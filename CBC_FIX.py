"""
============================================================
VALIDASI MILP: MT-CVRPTW-PG dengan PuLP + CBC Solver
============================================================
TA: Hanif Hisyam Ramadhan (2026) - ITS Surabaya
Pembimbing: Prof. Budi Santosa, Ph.D.

Model Matematis (Bab 3 TA):
  Min Z = C1 + C2 + C3                        (Pers. 3.3)
  C1 = Sum F * z_k                            (Pers. 3.4)
  C2 = Sum c * d_ij * x_ijkr                  (Pers. 3.5)
  C3 = Sum p * q_i * (1 - Q_ikr) * y_ikr      (Pers. 3.6)

Quality Decay (Pers. 3.1 & 3.2):
  alpha(T) = alpha_0 * exp(theta*T) / 60   [per menit]
  Q_jkr = Q_ikr * exp(-(alpha_arc*t_ij + alpha_node*s_i))

Sumber parameter decay: Gopalakrishnan et al. (2016) - ikan kembung:
  alpha_0=0.0321, theta=0.0654, T_arc=0 C, T_node=28 C
Parameter biaya:
  F = Q_k x 1.000 Rp, c = Rp 10.000/unit jarak, p = Rp 40.000/kg

Dataset: Solomon Benchmark | 7 pelanggan pertama (+ depot)
Solver : COIN-OR CBC via PuLP

Catatan metodologis:
  MILP/CBC bersifat DETERMINISTIK EKSAK -> tidak memerlukan
  repetisi statistik. N_REPETITIONS=1. Format output disamakan
  dengan tabel stabilitas meta-heuristik (kolom: Best_Z,
  Rata2_Biaya_Z, Rata2_Truk, Rata2_Waktu, Total_Valid, ARPD%,
  Alasan_Mayoritas_Gagal). Untuk N=1, Best_Z == Rata2_Biaya_Z
  dan ARPD = 0%.
============================================================
"""

import math
import time
import os
import pandas as pd
import pulp

# ---------------------------------------------------------
# BAGIAN 1: PARAMETER GLOBAL
# ---------------------------------------------------------
ALPHA_0    = 0.0321       # Gopalakrishnan et al. (2016)
THETA      = 0.0654       # Gopalakrishnan et al. (2016)
TEMP_ARC   = 0.0          # Suhu ruang pendingin (C)
TEMP_NODE  = 28.0         # Suhu saat bongkar muat (C)
FISH_PRICE = 40_000.0     # Rp/kg
Q_MIN      = 0.8          # Ambang batas kualitas minimum 80% (Pers. 3.14)
Q_0        = 1.0          # Kualitas awal 100% (Pers. 3.12)
VAR_COST   = 10_000.0     # c = Rp per unit jarak (Pers. 3.5)

# Pemisahan Big-M
BIG_M      = 1e6          # Konstanta Big-M untuk time propagation (satuan menit)
BIG_M_Q    = 2.0          # Konstanta Big-M untuk quality propagation (q dalam [0,1])


def alpha_decay(T: float) -> float:
    """Laju penurunan kualitas per menit pada suhu T (Pers. 3.1 / Lin et al., 2025)."""
    return (ALPHA_0 * math.exp(THETA * T)) / 60.0


ALPHA_ARC  = alpha_decay(TEMP_ARC)
ALPHA_NODE = alpha_decay(TEMP_NODE)


def decay_factor(t_ij: float, s_i: float) -> float:
    return math.exp(-(ALPHA_ARC * t_ij + ALPHA_NODE * s_i))


# ---------------------------------------------------------
# BAGIAN 2: PEMBACAAN DATA SOLOMON
# ---------------------------------------------------------
class Node:
    def __init__(self, cust_id, x, y, demand, ready_time, due_date, service_time):
        self.id           = int(cust_id)
        self.x            = float(x)
        self.y            = float(y)
        self.demand       = float(demand)
        self.ready_time   = float(ready_time)
        self.due_date     = float(due_date)
        self.service_time = float(service_time)


def load_solomon(file_path: str, num_customers: int = 7):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    capacity, max_veh = 200.0, 25
    for i, line in enumerate(lines):
        if 'NUMBER' in line.upper() and 'CAPACITY' in line.upper():
            vals     = lines[i + 1].strip().split()
            max_veh  = int(vals[0])
            capacity = float(vals[1])
            break

    start_idx = 0
    for i, line in enumerate(lines):
        if 'CUST NO.' in line.upper() or 'CUST NO' in line.upper():
            start_idx = i + 1
            break

    df = pd.read_csv(
        file_path, sep=r'\s+', skiprows=start_idx, header=None,
        names=['ID', 'X', 'Y', 'D', 'E', 'L', 'S']
    ).dropna().head(num_customers + 1)

    nodes = [Node(*row) for row in df.values]
    return nodes, capacity, max_veh


def euclidean(a: Node, b: Node) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ---------------------------------------------------------
# BAGIAN 3: MILP SOLVER
# ---------------------------------------------------------
def solve_milp(
    nodes:        list,
    capacity:     float,
    max_vehicles: int,
    max_trips:    int = 2,
    time_limit:   int = 90
) -> dict:
    n  = len(nodes) - 1
    V  = list(range(n + 1))
    N_ = list(range(1, n + 1))
    K  = list(range(max_vehicles))
    R  = list(range(max_trips))

    depot = nodes[0]
    T_MAX = depot.due_date
    FIXED = capacity * 1_000.0

    dist   = {(i, j): euclidean(nodes[i], nodes[j]) for i in V for j in V if i != j}
    tt     = dict(dist)
    df_arc = {(i, j): decay_factor(tt[(i, j)], nodes[i].service_time)
              for i in V for j in V if i != j}

    prob = pulp.LpProblem("MT_CVRPTW_PG", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", [(i, j, k, r) for i in V for j in V for k in K for r in R if i != j],
        cat='Binary')
    y = pulp.LpVariable.dicts(
        "y", [(i, k, r) for i in N_ for k in K for r in R], cat='Binary')
    z = pulp.LpVariable.dicts("z", K, cat='Binary')
    t_ = pulp.LpVariable.dicts(
        "t", [(i, k, r) for i in V for k in K for r in R], lowBound=0)
    Qvar = pulp.LpVariable.dicts(
        "Q", [(i, k, r) for i in V for k in K for r in R], lowBound=0, upBound=1)
    u = pulp.LpVariable.dicts(
        "u", [(i, k, r) for i in N_ for k in K for r in R], lowBound=0)

    w = pulp.LpVariable.dicts(
        "w", [(i, k, r) for i in N_ for k in K for r in R], lowBound=0, upBound=1)

    for i in N_:
        for k in K:
            for r in R:
                prob += w[(i, k, r)] <= Qvar[(i, k, r)]
                prob += w[(i, k, r)] <= y[(i, k, r)]
                prob += w[(i, k, r)] >= Qvar[(i, k, r)] + y[(i, k, r)] - 1

    qx = pulp.LpVariable.dicts(
        "qx",
        [(i, j, k, r) for i in V for j in N_ for k in K for r in R if i != j],
        lowBound=0, upBound=1)

    for i in V:
        for j in N_:
            if i == j:
                continue
            for k in K:
                for r in R:
                    prob += qx[(i, j, k, r)] <= Qvar[(i, k, r)]
                    prob += qx[(i, j, k, r)] <= x[(i, j, k, r)]
                    prob += qx[(i, j, k, r)] >= (
                        Qvar[(i, k, r)] + x[(i, j, k, r)] - 1)

    C1 = pulp.lpSum(FIXED * z[k] for k in K)
    C2 = pulp.lpSum(
        VAR_COST * dist[(i, j)] * x[(i, j, k, r)]
        for i in V for j in V for k in K for r in R if i != j)
    C3 = pulp.lpSum(
        FISH_PRICE * nodes[i].demand * (y[(i, k, r)] - w[(i, k, r)])
        for i in N_ for k in K for r in R)
    prob += C1 + C2 + C3, "Minimize_Z"

    for i in N_:
        prob += pulp.lpSum(y[(i, k, r)] for k in K for r in R) == 1

    for i in N_:
        for k in K:
            for r in R:
                prob += pulp.lpSum(x[(i, j, k, r)] for j in V if j != i) == y[(i, k, r)]
                prob += pulp.lpSum(x[(j, i, k, r)] for j in V if j != i) == y[(i, k, r)]

    for k in K:
        for r in R:
            prob += pulp.lpSum(nodes[i].demand * y[(i, k, r)] for i in N_) <= capacity

    for i in N_:
        for k in K:
            for r in R:
                prob += t_[(i, k, r)] >= nodes[i].ready_time * y[(i, k, r)]
                prob += t_[(i, k, r)] <= nodes[i].due_date   * y[(i, k, r)]

    for k in K:
        prob += pulp.lpSum(
            (tt[(i, j)] + nodes[i].service_time) * x[(i, j, k, r)]
            for i in V for j in V for r in R if i != j
        ) <= T_MAX

    for i in V:
        for j in N_:
            if i != j:
                for k in K:
                    for r in R:
                        prob += (t_[(j, k, r)] >=
                                 t_[(i, k, r)] + nodes[i].service_time + tt[(i, j)]
                                 - BIG_M * (1 - x[(i, j, k, r)]))

    for k in K:
        for r in R:
            prob += t_[(0, k, r)] == 0

    for k in K:
        for r in R:
            prob += Qvar[(0, k, r)] == Q_0

    for i in V:
        for j in N_:
            if i == j:
                continue
            d = df_arc[(i, j)]
            for k in K:
                for r in R:
                    prob += (Qvar[(j, k, r)] <=
                             d * qx[(i, j, k, r)] + BIG_M_Q * (1 - x[(i, j, k, r)]))

    for i in N_:
        for k in K:
            for r in R:
                prob += Qvar[(i, k, r)] >= Q_MIN * y[(i, k, r)]

    for k in K:
        for r in R[:-1]:
            prob += (pulp.lpSum(x[(0, j, k, r)]     for j in N_) >=
                     pulp.lpSum(x[(0, j, k, r + 1)] for j in N_))

    for i in N_:
        for k in K:
            for r in R:
                prob += u[(i, k, r)] >= nodes[i].demand
                prob += u[(i, k, r)] <= capacity
    for i in N_:
        for j in N_:
            if i != j:
                for k in K:
                    for r in R:
                        prob += (u[(i, k, r)] - u[(j, k, r)] + capacity * x[(i, j, k, r)]
                                 <= capacity - nodes[j].demand)

    for k in K:
        for j in N_:
            for r in R:
                prob += z[k] >= x[(0, j, k, r)]
    for k in K:
        prob += pulp.lpSum(y[(i, k, r)] for i in N_ for r in R) <= n * z[k]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    t0 = time.time()
    prob.solve(solver)
    solve_time = round(time.time() - t0, 3)

    status = pulp.LpStatus[prob.status]

    result = dict(Z=float('inf'), C1=0.0, C2=0.0, C3=0.0,
                  vehicles=0, solve_time=solve_time, status=status, routes=[])

    if pulp.value(prob.objective) is not None:
        result["Z"]  = round(pulp.value(prob.objective), 2)
        result["C1"] = round(pulp.value(C1), 2)
        result["C2"] = round(pulp.value(C2), 2)
        result["C3"] = round(pulp.value(C3), 2)
        result["vehicles"] = sum(
            1 for k in K if (pulp.value(z[k]) or 0) > 0.5
        )
        routes = []
        for k in K:
            for r in R:
                arcs = [(i, j) for i in V for j in V if i != j
                        and (pulp.value(x[(i, j, k, r)]) or 0) > 0.5]
                if arcs:
                    seq  = [0]
                    cur  = 0
                    used = set()
                    for _ in range(len(arcs)):
                        for (i, j) in arcs:
                            if i == cur and (i, j) not in used:
                                seq.append(j)
                                used.add((i, j))
                                cur = j
                                break
                    routes.append({"vehicle": k + 1, "trip": r + 1, "route": seq})
        result["routes"] = routes

    return result


# ---------------------------------------------------------
# BAGIAN 4: EKSEKUSI UTAMA (template format stabilitas)
# ---------------------------------------------------------
if __name__ == "__main__":

    DATASET_FILES = {
        "R101":  "R101.txt",
        "R111":  "R111.txt",
        "C101":  "C101.txt",
        "C105":  "C105.txt",
        "RC101": "RC101.txt",
        "RC105": "RC105.txt",
        "R201":  "R201.txt",
        "C201":  "C201.txt",
        "RC201": "RC201.txt",
        "R205":  "R205.txt",
    }

    NUM_CUSTOMERS = 7
    MAX_TRIPS     = 2
    MAX_VEHICLES  = 4
    TIME_LIMIT    = 90

    # MILP deterministik -> repetisi tidak diperlukan
    N_REPETITIONS = 1

    print("=" * 115)
    print(f"  VALIDASI EKSAK: MILP MT-CVRPTW-PG | {NUM_CUSTOMERS} PELANGGAN | CBC SOLVER")
    print(f"  Parameter   : Max kendaraan={MAX_VEHICLES}, Max trip/kendaraan={MAX_TRIPS}, TimeLimit={TIME_LIMIT}s")
    print(f"  alpha_arc   = {ALPHA_ARC:.6f}/mnt  |  alpha_node = {ALPHA_NODE:.6f}/mnt")
    print(f"  Q_min={Q_MIN*100:.0f}%  p=Rp{FISH_PRICE:,.0f}/kg  c=Rp{VAR_COST:,.0f}/unit jarak")
    print(f"  Catatan     : MILP deterministik -> 1 run per dataset (tidak perlu repetisi)")
    print("=" * 115)

    hasil_stabilitas = []

    for name, fpath in DATASET_FILES.items():
        if not os.path.exists(fpath):
            print(f"\n[SKIP] {name}: file '{fpath}' tidak ditemukan di folder Anda.")
            continue

        nodes, cap, max_veh = load_solomon(fpath, NUM_CUSTOMERS)
        FIXED_K = cap * 1_000
        print(f"\n> Memproses Dataset : {fpath}  (cap={cap:.0f}  F=Rp{FIXED_K:,.0f}  T_max={nodes[0].due_date})")

        res = solve_milp(
            nodes        = nodes,
            capacity     = cap,
            max_vehicles = MAX_VEHICLES,
            max_trips    = MAX_TRIPS,
            time_limit   = TIME_LIMIT
        )

        Z_str = f"Rp {res['Z']:>15,.2f}" if res['Z'] != float('inf') else "INFEASIBLE"
        print(f"   |- Status   : {res['status']}")
        print(f"   |- Z        : {Z_str}")
        print(f"   |- C1       : Rp {res['C1']:>12,.2f}")
        print(f"   |- C2       : Rp {res['C2']:>12,.2f}")
        print(f"   |- C3       : Rp {res['C3']:>12,.2f}")
        print(f"   |- Kendaraan: {res['vehicles']}")
        print(f"   |- Waktu CBC: {res['solve_time']} detik")
        if res['routes']:
            print("   |- Rute     :")
            for rt in res['routes']:
                print(f"      Kendaraan {rt['vehicle']} Trip {rt['trip']}: "
                      f"{' -> '.join(map(str, rt['route']))}")

        # Tentukan status validitas + alasan untuk kolom Alasan_Mayoritas_Gagal
        if res['Z'] == float('inf'):
            valid_count = 0
            alasan = "INFEASIBLE"
        elif res['status'] == "Optimal":
            valid_count = 1
            alasan = "Aman"
        else:
            # Feasible tapi tidak proven optimal (timeout CBC)
            valid_count = 1
            alasan = f"Not Proven ({res['status']})"

        # Untuk N=1: Best_Z == Rata2_Biaya_Z, ARPD = 0
        best_z = res['Z'] if res['Z'] != float('inf') else 0.0
        avg_z  = best_z
        arpd   = 0.00  # intra-instance ARPD untuk N=1 selalu 0

        print(f"   `- Ringkasan: Best Z = Rp {best_z:,.0f} | "
              f"Waktu = {res['solve_time']:.2f} s | Valid = {valid_count}/{N_REPETITIONS} | "
              f"Status = {alasan}")

        hasil_stabilitas.append({
            "Instance"              : name,
            "Best_Z (Rp)"           : best_z,
            "Rata2_Biaya_Z (Rp)"    : avg_z,
            "Rata2_Truk"            : float(res['vehicles']),
            "Rata2_Waktu (s)"       : float(res['solve_time']),
            "Total_Valid"           : f"{valid_count} / {N_REPETITIONS}",
            "ARPD (%)"              : arpd,
            "Alasan_Mayoritas_Gagal": alasan
        })

    # ---- TABEL RINGKASAN FINAL ----
    if hasil_stabilitas:
        df_stabilitas = pd.DataFrame(hasil_stabilitas)

        df_disp = df_stabilitas.copy()
        df_disp["Best_Z (Rp)"]        = df_disp["Best_Z (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
        df_disp["Rata2_Biaya_Z (Rp)"] = df_disp["Rata2_Biaya_Z (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
        df_disp["Rata2_Truk"]         = df_disp["Rata2_Truk"].apply(lambda x: f"{x:.1f}")
        df_disp["Rata2_Waktu (s)"]    = df_disp["Rata2_Waktu (s)"].apply(lambda x: f"{x:.2f} s")
        df_disp["ARPD (%)"]           = df_disp["ARPD (%)"].apply(lambda x: f"{x:.2f} %")

        print("\n" + "=" * 135)
        print(f"  TABEL RINGKASAN HASIL VALIDASI EKSAK MILP-CBC | {NUM_CUSTOMERS} PELANGGAN | {N_REPETITIONS} RUN")
        print("-" * 135)
        print(df_disp.to_string(index=False))
        print("=" * 135)

        nama_file = f"FinalRun_MILP_CBC_{NUM_CUSTOMERS}Cust_{N_REPETITIONS}Run.csv"
        df_stabilitas.to_csv(nama_file, index=False)
        print(f"\n[SUKSES] Hasil disimpan ke: '{nama_file}'")

        print("\n" + "=" * 115)
        print("KETERANGAN PARAMETER & METODOLOGI:")
        print(f"  alpha_0={ALPHA_0}  theta={THETA}  -> Gopalakrishnan et al. (2016)")
        print(f"  alpha_arc  = alpha({TEMP_ARC}C)/60  = {ALPHA_ARC:.6f} per menit  (dalam kendaraan)")
        print(f"  alpha_node = alpha({TEMP_NODE}C)/60 = {ALPHA_NODE:.6f} per menit  (bongkar muat)")
        print(f"  F = Q_k x 1.000 Rp per kendaraan")
        print(f"  MILP CBC: solver eksak DETERMINISTIK -> 1 run per dataset")
        print(f"  Best_Z == Rata2_Biaya_Z (N=1)  |  ARPD intra-instance = 0%")
        print(f"  Status 'Aman' = Optimal proven  |  'Not Proven' = feasible tapi CBC timeout")
        print(f"  Hasil Z di sini = BKS untuk pembanding ARPD-vs-BKS pada meta-heuristik")
        print("=" * 115)
