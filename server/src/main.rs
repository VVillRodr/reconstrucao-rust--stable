mod reconstruction;

use axum::{extract::State, http::StatusCode, response::Json, routing::{get, post}, Router};
// ALTERADO: Importações adicionais
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use ndarray::Array1;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    mpsc, Arc, Condvar, Mutex,
};
use std::thread;
use std::time::Duration;
use sysinfo::System;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use num_cpus;

struct ReconstructionJob {
    request: ReconstructionRequest,
    responder: oneshot::Sender<ReconstructionResult>,
}

struct ServerMetrics {
    queued_jobs: AtomicUsize,
    active_jobs: AtomicUsize,
    total_requests: AtomicU64,
    rejected_requests: AtomicU64,
    completed_requests: AtomicU64,
}

impl ServerMetrics {
    fn new() -> Self {
        Self {
            queued_jobs: AtomicUsize::new(0),
            active_jobs: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            completed_requests: AtomicU64::new(0),
        }
    }
}

struct ThreadSemaphore {
    state: Mutex<usize>,
    condvar: Condvar,
}

impl ThreadSemaphore {
    fn new(count: usize) -> Self {
        Self {
            state: Mutex::new(count),
            condvar: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let mut available = self.state.lock().unwrap();
        while *available == 0 {
            available = self.condvar.wait(available).unwrap();
        }
        *available -= 1;
    }

    fn release(&self) {
        let mut available = self.state.lock().unwrap();
        *available += 1;
        self.condvar.notify_one();
    }
}

struct AppState {
    sys: Mutex<System>,
    job_sender: mpsc::SyncSender<ReconstructionJob>,
    metrics: Arc<ServerMetrics>,
}

fn write_report_entry(result: &ReconstructionResult, image_filename: &str) -> std::io::Result<()> {
    const REPORT_FILE: &str = "reconstruction_report.csv";
    
    let file_exists = std::path::Path::new(REPORT_FILE).exists();
    let is_empty = if file_exists {
        std::fs::metadata(REPORT_FILE)?.len() == 0
    } else {
        true
    };

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(REPORT_FILE)?;

    if is_empty {
        // ALTERADO: Cabeçalho do CSV atualizado
        writeln!(file, "user_id,algorithm_id,start_time,end_time,reconstruction_ms,image_pixels,iterations,image_filename")?;
    }
    
    writeln!(
        file,
        "{},{},{},{},{},\"({},{})\",{},{}",
        result.user_id, // Uuid implementa Display, então .to_string() é chamado implicitamente
        result.algorithm_id,
        result.start_time.to_rfc3339(),
        result.end_time.to_rfc3339(),
        result.reconstruction_time_ms,
        result.image_pixels.0,
        result.image_pixels.1,
        result.iterations,
        image_filename
    )?;

    println!("[Worker] Entrada adicionada ao relatório: {}", REPORT_FILE);
    Ok(())
}


#[tokio::main]
async fn main() {
    env_logger::init();

    let mut sys = System::new_all();
    sys.refresh_memory();
    sys.refresh_cpu();

    let total_ram_mb = sys.total_memory() / 1024 / 1024;
    const MEMORY_UNIT_MB: u64 = 512;
    let total_memory_units = (total_ram_mb / MEMORY_UNIT_MB).max(1) as usize;
    let thread_limit = num_cpus::get().max(1);
    let queue_capacity = total_memory_units;
    
    // declaração do semáforo para limitar o número de threads ativas
    let thread_semaphore = Arc::new(ThreadSemaphore::new(thread_limit));
    println!(
        "[Servidor] Iniciando com {} thread(s) de execução e fila de {} jobs baseada em memória (~{} MB cada).",
        thread_limit,
        queue_capacity,
        MEMORY_UNIT_MB
    );

    
    //fila de jobs com capacidade dinamica a memoria disponível
    let (job_sender, job_receiver) = mpsc::sync_channel::<ReconstructionJob>(queue_capacity);
    
    
    
    
    let metrics = Arc::new(ServerMetrics::new());
    let job_receiver = Arc::new(Mutex::new(job_receiver));

    let shared_state = Arc::new(AppState {
        sys: Mutex::new(sys),
        job_sender,
        metrics: metrics.clone(),
    });

    // Relatório periódico de métricas.
    let logging_state = shared_state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            let mut sys = logging_state.sys.lock().unwrap();
            sys.refresh_cpu();
            sys.refresh_memory();

            let cpu = sys.global_cpu_info().cpu_usage();
            let used_mb = sys.used_memory() / 1024 / 1024;
            let total_mb = sys.total_memory() / 1024 / 1024;
            let queued = logging_state.metrics.queued_jobs.load(Ordering::Relaxed);
            let active = logging_state.metrics.active_jobs.load(Ordering::Relaxed);
            let total = logging_state.metrics.total_requests.load(Ordering::Relaxed);
            let rejected = logging_state.metrics.rejected_requests.load(Ordering::Relaxed);
            let completed = logging_state.metrics.completed_requests.load(Ordering::Relaxed);

            println!(
                "[Status] CPU: {cpu:.1}% | Memória: {used_mb}/{total_mb} MB | Fila: {queued} jobs | Ativos: {active} | Total: {total} | Rejeitados: {rejected} | Concluídos: {completed}"
            );
        }
    });


    //o dispatcher é responsável por receber jobs da fila e criar threads de worker para processá-los
    let dispatcher_receiver = job_receiver.clone();
    let dispatcher_metrics = metrics.clone();
    let dispatcher_semaphore = thread_semaphore.clone();
    thread::spawn(move || {
        println!("[Dispatcher] thread iniciada.");
        let mut job_id = 0;
        loop {
            let job = dispatcher_receiver.lock().unwrap().recv();//obtem o próximo job da fila
            let job = match job {
                Ok(job) => job,
                Err(_) => break,
            };

            dispatcher_semaphore.acquire(); // adquire o semáforo antes de iniciar um novo job
            //atualiza métricas
            dispatcher_metrics.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            dispatcher_metrics.active_jobs.fetch_add(1, Ordering::Relaxed);
            job_id += 1;

            let metrics = dispatcher_metrics.clone();
            let thread_semaphore = dispatcher_semaphore.clone();


            //cria a thread de worker para processar o job
            thread::spawn(move || {
                let worker_id = job_id;
                println!("[Worker-{}] job iniciado.", worker_id);

                let request = job.request;
            // Determina as dimensões da imagem com base no model_id   
                let image_pixels = match request.model_id.as_str() {
                    "30x30" => (30, 30),
                    "60x60" => (60, 60),
                    other => {
                        eprintln!("[Worker-{}] ERRO: model_id '{}' não suportado.", worker_id, other);
                        let _ = job.responder.send(ReconstructionResult::new_error(request.user_id, request.algorithm_id));
                        metrics.active_jobs.fetch_sub(1, Ordering::Relaxed);
                        thread_semaphore.release();
                        return;
                    }
                };
                // lê a matriz de reconstrção correspondente ao modelo 
                let h_file = format!("H-{}.csv", request.model_id);
                let s_samples = request.g.len();
                let n_pixels = image_pixels.0 * image_pixels.1;

                let algorithm_id = request.algorithm_id.clone();
                let user_id = request.user_id;
                let g_vec = request.g;

                let h_matrix = match reconstruction::read_h_matrix_from_csv(&h_file, s_samples, n_pixels) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("[Worker-{}] ERRO: Falha ao carregar o arquivo H: {}", worker_id, e);
                        let _ = job.responder.send(ReconstructionResult::new_error(user_id, algorithm_id));
                        metrics.active_jobs.fetch_sub(1, Ordering::Relaxed);
                        thread_semaphore.release();
                        return;
                    }
                };

                // Executa o algoritmo de reconstrução
                let result = match algorithm_id.as_str() {
                    "CGNR" => reconstruction::execute_cgnr(
                        &algorithm_id,
                        user_id,
                        &h_matrix,
                        &Array1::from(g_vec.clone()),
                        image_pixels,
                    ),
                    "CGNE" => reconstruction::execute_cgne(
                        &algorithm_id,
                        user_id,
                        &h_matrix,
                        &Array1::from(g_vec.clone()),
                        image_pixels,
                    ),
                    other => {
                        eprintln!("[Worker-{}] ERRO: algorithm_id '{}' não suportado.", worker_id, other);
                        let error_result = ReconstructionResult::new_error(user_id, algorithm_id.clone());
                        if job.responder.send(error_result).is_err() {
                            eprintln!("[Worker-{}] Falha ao enviar resposta de erro.", worker_id);
                        }
                        metrics.active_jobs.fetch_sub(1, Ordering::Relaxed);
                        thread_semaphore.release();
                        return;
                    }
                };
                //salva a imagem reconstruida
                let image_filename = match reconstruction::save_image(&result) {
                    Ok(name) => name,
                    Err(e) => {
                        eprintln!("[Worker-{}] Erro ao salvar imagem: {}", worker_id, e);
                        String::from("save_failed")
                    }
                };

                if let Err(e) = write_report_entry(&result, &image_filename) {
                    eprintln!("[Worker-{}] ERRO: Falha ao escrever no arquivo de relatório: {}", worker_id, e);
                }

                if job.responder.send(result).is_err() {
                    eprintln!("[Worker-{}] Falha ao enviar resposta. O cliente provavelmente desistiu.", worker_id);
                }

                metrics.active_jobs.fetch_sub(1, Ordering::Relaxed);
                metrics.completed_requests.fetch_add(1, Ordering::Relaxed);
                thread_semaphore.release(); //libera o semáforo após concluir o job, assim, liberando a thread para outro job

                println!("[Worker-{}] job finalizado.", worker_id);
            });
        }
        println!("[Dispatcher] thread finalizada.");
    });


    //aqui expomos as rotas do servidor. o /reconstruct será chamado pelo cliente 
    // para enviar os jobs de reconstrução, e o /status para monitoramento do servidor
    let app = Router::new()
        .route("/reconstruct", post(handle_reconstruction))
        .route("/status", get(handle_status))
        .with_state(shared_state);

    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("[Servidor] Ouvindo em http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}

//essa função serve para lidar com a requisição de reconstrução, enfileirando o job e aguardando a resposta do worker
async fn handle_reconstruction(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ReconstructionRequest>,
) -> Result<Json<ReconstructionResult>, StatusCode> {
    let user_id_for_log = payload.user_id;
    state.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

    if payload.model_id != "30x30" && payload.model_id != "60x60" {
        eprintln!("[Servidor] ERRO: model_id '{}' não suportado.", payload.model_id);
        state.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
        return Err(StatusCode::BAD_REQUEST);
    }

    let (response_sender, response_receiver) = oneshot::channel();
    let job = ReconstructionJob {
        request: payload,
        responder: response_sender,
    };

    //tenta enfileirar o job, se a fila estiver cheia, rejeita a requisição
    match state.job_sender.try_send(job) {
        Ok(()) => {
            state.metrics.queued_jobs.fetch_add(1, Ordering::Relaxed);
            println!("[Servidor] Requisição do usuário {} enfileirada com sucesso.", user_id_for_log);
        }
        Err(err) => {
            let _job = match err {
                mpsc::TrySendError::Full(job) => job,
                mpsc::TrySendError::Disconnected(job) => job,
            };
            state.metrics.rejected_requests.fetch_add(1, Ordering::Relaxed);
            eprintln!("[Servidor] Rejeitando requisição do usuário {}: fila cheia.", user_id_for_log);
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    match response_receiver.await {
        Ok(result) => {
            if result.iterations == 0 && result.reconstruction_time_ms == 0 && result.f.is_empty() {
                eprintln!("[Servidor] A tarefa para o usuário {} resultou em um erro no worker.", user_id_for_log);
                Err(StatusCode::BAD_REQUEST)
            } else {
                Ok(Json(result))
            }
        }
        Err(_) => {
            eprintln!("[Servidor] A tarefa para o usuário {} falhou (canal de resposta fechado).", user_id_for_log);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_status(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ServerStatus>) {
    let mut sys = state.sys.lock().unwrap();
    sys.refresh_cpu();
    sys.refresh_memory();
    let status = ServerStatus {
        cpu_usage: sys.global_cpu_info().cpu_usage(),
        memory_usage_mb: sys.used_memory() / 1024 / 1024,
        total_memory_mb: sys.total_memory() / 1024 / 1024,
        queued_jobs: state.metrics.queued_jobs.load(Ordering::Relaxed),
        active_jobs: state.metrics.active_jobs.load(Ordering::Relaxed),
        total_requests: state.metrics.total_requests.load(Ordering::Relaxed),
        rejected_requests: state.metrics.rejected_requests.load(Ordering::Relaxed),
        completed_requests: state.metrics.completed_requests.load(Ordering::Relaxed),
    };
    (StatusCode::OK, Json(status))
}