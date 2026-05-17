"""
============================================================
VALIDASI METAHEURISTIK: Hybrid GA + 2-Opt (7 Pelanggan)
============================================================
TA: Hanif Hisyam Ramadhan (2026) - ITS Surabaya
Pembimbing: Prof. Budi Santosa, Ph.D.

Versi 10-repetisi paralel: setiap dataset dijalankan N_REPETITIONS
kali secara paralel (ProcessPoolExecutor). Ringkasan stabilitas
dicetak dengan kolom: Best_Z, Rata2_Biaya_Z, Rata2_Truk,
Rata2_Waktu, Total_Valid, ARPD (%), Alasan_Mayoritas_Gagal.

Numba JIT pada evaluator + 2-Opt. Model penalti dipertahankan:
SATU big_M jika (telat ATAU busuk) - identik dengan kode asli.
Bitflag pelanggaran dilacak terpisah hanya untuk pelaporan
"alasan", tidak mempengaruhi nilai penalti maupun Z.
============================================================
"""

import math
import numpy as np
import pandas as pd
import random
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

# ---------------------------------------------------------
# JIT-COMPILED CORE: EVALUATOR + 2-OPT
# Model penalti 7C: 1x big_M jika (telat OR busuk).
# viol_flags hanya untuk pelaporan (tidak mengubah penalti/Z).
# Buffer trip_flat + trip_starts untuk routes_record.
# ---------------------------------------------------------
@njit(cache=True, fastmath=False)
def _calc_cost_jit(chrom,
                   demands, ready_times, due_dates, service_times,
                   distance_matrix, time_matrix,
                   Q_k, max_veh, fixed_cost, var_cost, big_M, t_max,
                   fish_price, q_min, alpha_arc, alpha_node):
    n_chrom = chrom.shape[0]

    trip_flat   = np.empty(n_chrom, dtype=np.int64)
    trip_starts = np.empty(n_chrom + 2, dtype=np.int64)
    trip_flat_len = 0
    n_trips = 1
    trip_starts[0] = 0

    vehicles_used = 1
    current_vehicle_time = 0.0
    current_node  = 0
    current_time  = 0.0
    current_load  = 0.0
    current_quality = 1.0
    total_distance  = 0.0
    quality_penalty_cost = 0.0
    penalty_violation    = 0.0

    # Bitflag pelaporan: 1=Telat Waktu, 2=Ikan Busuk, 4=Over Armada
    viol_flags = 0

    for k in range(n_chrom):
        cust_id = chrom[k]
        t_ij = time_matrix[current_node, cust_id]
        if current_node != 0:
            s_i = service_times[current_node]
        else:
            s_i = 0.0

        cust_ready  = ready_times[cust_id]
        cust_due    = due_dates[cust_id]
        cust_demand = demands[cust_id]
        cust_svc    = service_times[cust_id]

        arrival_time = current_time + t_ij
        wait_time = cust_ready - arrival_time
        if wait_time < 0.0:
            wait_time = 0.0
        service_start = arrival_time + wait_time

        decay_factor = math.exp(-(alpha_arc * t_ij + alpha_node * s_i))
        tentative_quality = current_quality * decay_factor

        if (current_load + cust_demand > Q_k) or (service_start > cust_due) or (tentative_quality < q_min):
            total_distance       += distance_matrix[current_node, 0]
            current_vehicle_time += distance_matrix[current_node, 0]

            # Tutup trip lama
            trip_starts[n_trips] = trip_flat_len
            n_trips += 1

            t_0j = time_matrix[0, cust_id]
            est_new_trip_time = t_0j + cust_svc + time_matrix[cust_id, 0]

            if (current_vehicle_time + est_new_trip_time > t_max) or (current_vehicle_time + t_0j > cust_due):
                vehicles_used += 1
                current_vehicle_time = 0.0

            current_node    = 0
            current_time    = current_vehicle_time
            current_load    = 0.0
            current_quality = 1.0

            arrival_time  = current_time + t_0j
            w2 = cust_ready - arrival_time
            if w2 < 0.0:
                w2 = 0.0
            service_start = arrival_time + w2
            tentative_quality = math.exp(-(alpha_arc * t_0j))

        # Penalti versi 7C: SATU big_M jika telat ATAU busuk.
        # Bitflag dilacak terpisah (tidak mempengaruhi penalti).
        if (service_start > cust_due) or (tentative_quality < q_min):
            penalty_violation += big_M
            if service_start > cust_due:
                viol_flags |= 1
            if tentative_quality < q_min:
                viol_flags |= 2

        current_load   += cust_demand
        current_time    = service_start + cust_svc
        total_distance += distance_matrix[current_node, cust_id]
        current_node    = cust_id
        current_vehicle_time = current_time
        current_quality = tentative_quality
        quality_penalty_cost += fish_price * cust_demand * (1.0 - current_quality)

        trip_flat[trip_flat_len] = cust_id
        trip_flat_len += 1

    total_distance += distance_matrix[current_node, 0]
    trip_starts[n_trips] = trip_flat_len

    c1_cost = vehicles_used * fixed_cost
    c2_cost = total_distance * var_cost

    if vehicles_used > max_veh:
        penalty_violation += big_M * (vehicles_used - max_veh)
        viol_flags |= 4

    real_z      = c1_cost + c2_cost + quality_penalty_cost
    penalized_z = real_z + penalty_violation
    fitness     = 1.0 / (1.0 + penalized_z)

    # Salin ke array berukuran eksak
    out_trip_flat = np.empty(trip_flat_len, dtype=np.int64)
    for i in range(trip_flat_len):
        out_trip_flat[i] = trip_flat[i]
    out_trip_starts = np.empty(n_trips + 1, dtype=np.int64)
    for i in range(n_trips + 1):
        out_trip_starts[i] = trip_starts[i]

    return (real_z, penalized_z, fitness, vehicles_used,
            c1_cost, c2_cost, quality_penalty_cost, penalty_violation,
            out_trip_flat, out_trip_starts, viol_flags)


@njit(cache=True, fastmath=False)
def _apply_2opt_jit(chrom,
                    demands, ready_times, due_dates, service_times,
                    distance_matrix, time_matrix,
                    Q_k, max_veh, fixed_cost, var_cost, big_M, t_max,
                    fish_price, q_min, alpha_arc, alpha_node):
    best_route = chrom.copy()
    res = _calc_cost_jit(best_route, demands, ready_times, due_dates, service_times,
                         distance_matrix, time_matrix,
                         Q_k, max_veh, fixed_cost, var_cost, big_M, t_max,
                         fish_price, q_min, alpha_arc, alpha_node)
    best_pen_z = res[1]
    n = best_route.shape[0]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                left = i
                right = j - 1
                while left < right:
                    tmp = best_route[left]
                    best_route[left]  = best_route[right]
                    best_route[right] = tmp
                    left  += 1
                    right -= 1

                res2 = _calc_cost_jit(best_route, demands, ready_times, due_dates, service_times,
                                      distance_matrix, time_matrix,
                                      Q_k, max_veh, fixed_cost, var_cost, big_M, t_max,
                                      fish_price, q_min, alpha_arc, alpha_node)
                new_pen_z = res2[1]

                if new_pen_z < best_pen_z:
                    best_pen_z = new_pen_z
                    improved = True
                    break
                else:
                    left = i
                    right = j - 1
                    while left < right:
                        tmp = best_route[left]
                        best_route[left]  = best_route[right]
                        best_route[right] = tmp
                        left  += 1
                        right -= 1
            if improved:
                break
    return best_route


# ---------------------------------------------------------
# BAGIAN 1: STRUKTUR DATA & ENVIRONMENT
# ---------------------------------------------------------
class Customer:
    def __init__(self, cust_id, x, y, demand, ready_time, due_date, service_time):
        self.id = int(cust_id)
        self.x = float(x)
        self.y = float(y)
        self.demand = float(demand)
        self.ready_time = float(ready_time)
        self.due_date = float(due_date)
        self.service_time = float(service_time)

class VRPEnvironment:
    def __init__(self, file_path, num_customers=7):
        self.file_path = file_path
        self.num_customers = num_customers
        self.nodes = []
        self.max_vehicles_dataset = 0
        self.vehicle_capacity_dataset = 0.0

        self.alpha_0 = 0.0321
        self.theta = 0.0654
        self.temp_arc = 0.0
        self.temp_node = 28.0
        self.alpha_arc = self._calculate_alpha(self.temp_arc)
        self.alpha_node = self._calculate_alpha(self.temp_node)

        self.fish_price = 40000.0
        self.q_min = 0.8

        self._load_solomon_data()
        self._calculate_matrices()

    def _calculate_alpha(self, temperature):
        return (self.alpha_0 * math.exp(self.theta * temperature)) / 60.0

    def _load_solomon_data(self):
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if 'NUMBER' in line.upper() and 'CAPACITY' in line.upper():
                    vals = lines[i+1].strip().split()
                    self.max_vehicles_dataset = int(vals[0])
                    self.vehicle_capacity_dataset = float(vals[1])
                    break

            start_idx = 0
            for i, line in enumerate(lines):
                if 'CUST NO.' in line.upper() or 'CUST NO' in line.upper():
                    start_idx = i + 1
                    break

            df = pd.read_csv(
                self.file_path, sep=r'\s+', skiprows=start_idx, header=None,
                names=['CUST_NO', 'X', 'Y', 'DEMAND', 'READY_TIME', 'DUE_DATE', 'SERVICE_TIME']
            )
            df = df.dropna().head(self.num_customers + 1)

            for _, row in df.iterrows():
                self.nodes.append(Customer(
                    row['CUST_NO'], row['X'], row['Y'], row['DEMAND'],
                    row['READY_TIME'], row['DUE_DATE'], row['SERVICE_TIME']
                ))
        except Exception as e:
            pass

    def _calculate_matrices(self):
        n = len(self.nodes)
        self.distance_matrix = np.zeros((n, n))
        self.time_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = math.sqrt((self.nodes[i].x - self.nodes[j].x)**2 + (self.nodes[i].y - self.nodes[j].y)**2)
                    self.distance_matrix[i][j] = dist
                    self.time_matrix[i][j] = dist

    def get_depot(self):
        return self.nodes[0]

# ---------------------------------------------------------
# BAGIAN 2: EVALUATOR (THIN WRAPPER ATAS JIT)
# Return 10-tuple: (real_z, pz, fit, veh, c1, c2, qpc, pv,
#                   routes_record, alasan_str)
# ---------------------------------------------------------
class RouteEvaluator:
    def __init__(self, env, var_cost):
        self.env = env
        self.Q_k = float(env.vehicle_capacity_dataset)
        self.max_veh = int(env.max_vehicles_dataset)
        self.fixed_cost = float(self.Q_k * 1000)
        self.var_cost = float(var_cost)
        self.big_M = 1e9
        self.t_max = float(env.get_depot().due_date)

        self.demands       = np.array([c.demand for c in env.nodes],       dtype=np.float64)
        self.ready_times   = np.array([c.ready_time for c in env.nodes],   dtype=np.float64)
        self.due_dates     = np.array([c.due_date for c in env.nodes],     dtype=np.float64)
        self.service_times = np.array([c.service_time for c in env.nodes], dtype=np.float64)
        self.distance_matrix = np.ascontiguousarray(env.distance_matrix, dtype=np.float64)
        self.time_matrix     = np.ascontiguousarray(env.time_matrix,     dtype=np.float64)

        self.fish_price = float(env.fish_price)
        self.q_min      = float(env.q_min)
        self.alpha_arc  = float(env.alpha_arc)
        self.alpha_node = float(env.alpha_node)

    def _to_arr(self, chromosome):
        if isinstance(chromosome, np.ndarray):
            return chromosome if chromosome.dtype == np.int64 else chromosome.astype(np.int64)
        return np.asarray(chromosome, dtype=np.int64)

    def calculate_cost(self, chromosome):
        chrom_arr = self._to_arr(chromosome)
        (real_z, penalized_z, fitness, vehicles_used,
         c1_cost, c2_cost, qpc, pv,
         trip_flat, trip_starts, viol_flags) = _calc_cost_jit(
            chrom_arr,
            self.demands, self.ready_times, self.due_dates, self.service_times,
            self.distance_matrix, self.time_matrix,
            self.Q_k, self.max_veh, self.fixed_cost, self.var_cost, self.big_M, self.t_max,
            self.fish_price, self.q_min, self.alpha_arc, self.alpha_node)

        routes_record = []
        n_trips = len(trip_starts) - 1
        for k in range(n_trips):
            start = int(trip_starts[k])
            end   = int(trip_starts[k+1])
            trip = [0] + [int(x) for x in trip_flat[start:end]] + [0]
            routes_record.append(trip)

        labels = []
        if viol_flags & 1: labels.append("Telat Waktu")
        if viol_flags & 2: labels.append("Ikan Busuk (<80%)")
        if viol_flags & 4: labels.append("Over Armada")
        alasan_str = ", ".join(labels) if labels else "Valid"

        return (real_z, penalized_z, fitness, int(vehicles_used),
                c1_cost, c2_cost, qpc, pv, routes_record, alasan_str)

    def apply_2opt_arr(self, chromosome):
        chrom_arr = self._to_arr(chromosome)
        return _apply_2opt_jit(
            chrom_arr,
            self.demands, self.ready_times, self.due_dates, self.service_times,
            self.distance_matrix, self.time_matrix,
            self.Q_k, self.max_veh, self.fixed_cost, self.var_cost, self.big_M, self.t_max,
            self.fish_price, self.q_min, self.alpha_arc, self.alpha_node)

# ---------------------------------------------------------
# BAGIAN 3: HYBRID GA + 2-OPT
# ---------------------------------------------------------
class HybridGA:
    def __init__(self, env, evaluator, pop_size=100, max_gen=100, p_m=0.3, elitism_rate=0.1, p_2opt=0.15):
        self.env = env
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_m = p_m
        self.p_2opt = p_2opt
        self.num_elites = max(1, int(elitism_rate * pop_size))
        self.num_customers = env.num_customers

    def apply_2opt(self, chromosome):
        opt_arr = self.evaluator.apply_2opt_arr(chromosome)
        return opt_arr.tolist()

    def run(self):
        population = []
        base_chrom = list(range(1, self.num_customers + 1))
        num_enhanced = max(1, int(0.1 * self.pop_size))

        for i in range(self.pop_size):
            chrom = base_chrom.copy()
            random.shuffle(chrom)
            population.append(self.apply_2opt(chrom) if i < num_enhanced else chrom)

        best_details = None
        best_route_chrom = None
        best_penalized_cost = float('inf')

        for gen in range(self.max_gen):
            pop_fitnesses = []
            for chrom in population:
                res = self.evaluator.calculate_cost(chrom)
                pop_fitnesses.append(res[2])
                if res[1] < best_penalized_cost:
                    best_penalized_cost = res[1]
                    best_details = res
                    best_route_chrom = chrom.copy()

            sorted_indices = np.argsort(pop_fitnesses)[::-1]
            new_population = []

            for i in sorted_indices[:self.num_elites]:
                elite_chrom = population[i].copy()
                optimized_elite = self.apply_2opt(elite_chrom)
                new_population.append(optimized_elite)

            total_fitness = sum(pop_fitnesses)
            while len(new_population) < self.pop_size:
                pick = random.uniform(0, total_fitness)
                current = 0
                parent = population[-1]
                for i, f in enumerate(pop_fitnesses):
                    current += f
                    if current >= pick:
                        parent = population[i]
                        break

                offspring = parent.copy()

                if random.random() < self.p_m:
                    if random.random() < 0.5:
                        idx = random.randint(0, len(offspring)-1)
                        val = offspring.pop(idx)
                        offspring.insert(random.randint(0, len(offspring)), val)
                    else:
                        idx1, idx2 = random.sample(range(len(offspring)), 2)
                        offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]

                if random.random() < self.p_2opt:
                    offspring = self.apply_2opt(offspring)

                new_population.append(offspring)
            population = new_population

        return best_route_chrom, best_details


# ---------------------------------------------------------
# WORKER UNTUK MULTIPROCESSING
# ---------------------------------------------------------
def single_run_worker(args):
    file_name, run_ke, num_customers, pop_size, max_gen, p_m, elitism, p_2opt, var_cost = args

    random.seed(run_ke * 1000 + int(time.time() * 1000) % 100000)
    np.random.seed(run_ke * 1000 + int(time.time() * 1000) % 100000)

    env = VRPEnvironment(file_path=file_name, num_customers=num_customers)
    if len(env.nodes) == 0:
        return run_ke, None, None, 0.0

    evaluator = RouteEvaluator(env, var_cost=var_cost)
    solver = HybridGA(env, evaluator,
                      pop_size=pop_size, max_gen=max_gen,
                      p_m=p_m, elitism_rate=elitism, p_2opt=p_2opt)

    t_start = time.time()
    best_route_chrom, best_details = solver.run()
    t_end = time.time()

    return run_ke, best_route_chrom, best_details, round(t_end - t_start, 3)


# ---------------------------------------------------------
# BAGIAN 4: EKSEKUSI UTAMA (10 DATASET x 10 REPETISI PARALEL)
# ---------------------------------------------------------
if __name__ == "__main__":

    DATASET_FILES = [
        "R101.txt", "R111.txt", "C101.txt", "C105.txt", "RC101.txt",
        "RC105.txt", "R201.txt", "C201.txt", "RC201.txt", "R205.txt"
    ]

    NUM_CUSTOMERS = 7
    N_REPETITIONS = 10
    VAR_COST = 10000

    PM       = 0.1
    ELITISM  = 0.20
    P_2OPT   = 0.3
    POP_SIZE = 100
    MAX_GEN  = 100

    N_WORKERS = os.cpu_count()

    # KUNCI JAWABAN (Best Known Solution) DARI HASIL EKSAK CBC
    BKS_EXACT = {
        "R101": 2472852.53, "R111": 1958069.39, "C101": 2715870.13,
        "C105": 2715870.13, "RC101": 2218959.69, "RC105": 2224621.52,
        "R201": 2987039.12, "C201": 4163402.66, "RC201": 3018959.69,
        "R205": 2834978.65
    }

    print("=" * 115)
    print(f"  VALIDASI METAHEURISTIK: HYBRID GA + 2-OPT | {NUM_CUSTOMERS} PELANGGAN | PARALEL")
    print(f"  Parameter : Pop={POP_SIZE}, Gen={MAX_GEN}, Pm={PM}, Elite={ELITISM}, P_2opt={P_2OPT}")
    print(f"  Repetisi  : {N_REPETITIONS}x per Dataset")
    print(f"  CPU Aktif : {N_WORKERS} core (semua terpakai)")
    print("=" * 115)

    hasil_stabilitas = []

    for file_name in DATASET_FILES:
        name = file_name.replace(".txt", "")
        if not os.path.exists(file_name):
            print(f"\n[SKIP] {name}: file '{file_name}' tidak ditemukan.")
            continue

        print(f"\n> Memproses Dataset : {file_name}  ({N_REPETITIONS} run paralel) ...")

        args_list = [
            (file_name, run_ke, NUM_CUSTOMERS, POP_SIZE, MAX_GEN,
             PM, ELITISM, P_2OPT, VAR_COST)
            for run_ke in range(1, N_REPETITIONS + 1)
        ]

        raw_results = {}
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_run = {executor.submit(single_run_worker, args): args[1] for args in args_list}
            for future in as_completed(future_to_run):
                try:
                    run_ke, best_route_chrom, best_details, waktu = future.result()
                    raw_results[run_ke] = (best_route_chrom, best_details, waktu)
                except Exception as exc:
                    run_ke = future_to_run[future]
                    print(f"   [ERROR] Run {run_ke} gagal: {exc}")
                    raw_results[run_ke] = (None, None, 0.0)

        list_z      = []
        list_truk   = []
        list_waktu  = []
        count_valid = 0
        alasan_gabungan = set()

        for run_ke in sorted(raw_results.keys()):
            best_route_chrom, best_details, waktu_eksekusi = raw_results[run_ke]
            if best_details is None:
                print(f"   |- Run {run_ke:2d}: [GAGAL]")
                continue

            real_z       = best_details[0]
            armada       = best_details[3]
            penalty      = best_details[7]
            alasan_gagal = best_details[9]

            list_z.append(real_z)
            list_truk.append(armada)
            list_waktu.append(waktu_eksekusi)

            if penalty == 0:
                count_valid += 1
                status_cetak = "VALID"
            else:
                status_cetak = f"TDK VALID ({alasan_gagal})"
                alasan_gabungan.add(alasan_gagal)

            print(f"   |- Run {run_ke:2d}: Z = Rp {real_z:>13,.0f} | "
                  f"Truk = {armada} | Wkt = {waktu_eksekusi:>6.3f} s | Status: {status_cetak}")

        if len(list_z) == 0:
            continue

        best_z_lokal    = min(list_z)
        avg_z           = float(np.mean(list_z))
        avg_truk        = float(np.mean(list_truk))
        avg_waktu       = float(np.mean(list_waktu))
        rpd_values      = [((z - best_z_lokal) / best_z_lokal) * 100 for z in list_z]
        arpd_stabilitas = float(np.mean(rpd_values))
        ringkasan       = "Aman" if count_valid == len(list_z) else " / ".join(list(alasan_gabungan))

        # ARPD vs BKS (validasi terhadap CBC) untuk ditampilkan dalam log
        bks_lokal = BKS_EXACT.get(name, None)
        if bks_lokal is not None:
            arpd_vs_bks = ((best_z_lokal - bks_lokal) / bks_lokal) * 100
            print(f"   `- Ringkasan : Best Z = Rp {best_z_lokal:,.0f} | "
                  f"Avg Wkt = {avg_waktu:.3f} s | Valid = {count_valid}/{len(list_z)} | "
                  f"ARPD vs BKS = {arpd_vs_bks:.2f}%")
        else:
            print(f"   `- Ringkasan : Best Z = Rp {best_z_lokal:,.0f} | "
                  f"Avg Wkt = {avg_waktu:.3f} s | Valid = {count_valid}/{len(list_z)}")

        hasil_stabilitas.append({
            "Instance"              : name,
            "Best_Z (Rp)"           : best_z_lokal,
            "Rata2_Biaya_Z (Rp)"    : avg_z,
            "Rata2_Truk"            : avg_truk,
            "Rata2_Waktu (s)"       : avg_waktu,
            "Total_Valid"           : f"{count_valid} / {len(list_z)}",
            "ARPD (%)"              : arpd_stabilitas,
            "Alasan_Mayoritas_Gagal": ringkasan
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
        print(f"  TABEL RINGKASAN UJI STABILITAS HYBRID GA+2-OPT | {NUM_CUSTOMERS} PELANGGAN | {N_REPETITIONS} REPETISI")
        print("-" * 135)
        print(df_disp.to_string(index=False))
        print("=" * 135)

        nama_file = f"FinalRun_HybridGA_2Opt_{NUM_CUSTOMERS}Cust_{N_REPETITIONS}Rep_Parallel.csv"
        df_stabilitas.to_csv(nama_file, index=False)
        print(f"\n[SUKSES] Hasil disimpan ke: '{nama_file}'")
