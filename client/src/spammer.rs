use common::ReconstructionRequest;
use ndarray::Array1;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::interval;
use uuid::Uuid;
use rand::Rng;

const SERVER_URL: &str = "http://127.0.0.1:3000";

fn resolve_csv_path(file_path: &str) -> PathBuf {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    workspace_root.join(file_path)
}

/// Função para ler o vetor 'g' de um arquivo CSV.
fn read_g_vector_from_csv(file_path: &str) -> Result<Array1<f64>, String> {
    let full_path = resolve_csv_path(file_path);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&full_path)
        .map_err(|e| format!("{} (path: {})", e.to_string(), full_path.display()))?;

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let value: f64 = record[0].trim().parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        data.push(value);
    }
    Ok(Array1::from(data))
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let client = reqwest::Client::new();
    let mut ticker = interval(Duration::from_millis(300));

    let signal_files = [
        "g-30x30-1.csv",
        "g-30x30-2.csv",
        "g-30x30-3.csv",
        "G-60x60-1.csv",
        "G-60x60-2.csv",
        "G-60x60-3.csv",
    ];

    println!("[Spammer] Iniciando spam de 200 requisições por minuto...");

    let mut count: u64 = 0;
    loop {
        ticker.tick().await;
        count += 1;

        let random_index = rand::thread_rng().gen_range(0..signal_files.len());
        let signal_file_to_load = signal_files[random_index];

        let g_vector = match read_g_vector_from_csv(signal_file_to_load) {
            Ok(g) => g,
            Err(error_message) => {
                eprintln!("[Spammer] ERRO ao ler '{}': {}", signal_file_to_load, error_message);
                continue;
            }
        };

        let model_id = if signal_file_to_load.contains("60x60") {
            "60x60"
        } else {
            "30x30"
        };

        let algorithms = ["CGNR", "CGNE"];

        for algorithm_id in algorithms {
            let request = ReconstructionRequest {
                user_id: Uuid::new_v4(),
                algorithm_id: algorithm_id.to_string(),
                model_id: model_id.to_string(),
                g: g_vector.to_vec(),
            };

            let client = client.clone();
            let request_count = count;
            let algorithm_name = algorithm_id.to_string();
            tokio::spawn(async move {
                match client
                    .post(format!("{}/reconstruct", SERVER_URL))
                    .json(&request)
                    .send()
                    .await
                {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            println!("[Spammer] Requisição #{} (algoritmo {}) enviada com sucesso.", request_count, algorithm_name);
                        } else {
                            let text = response.text().await.unwrap_or_default();
                            eprintln!("[Spammer] Requisição #{} (algoritmo {}) falhou: {} - {}", request_count, algorithm_name, status, text);
                        }
                    }
                    Err(e) => eprintln!("[Spammer] Falha ao enviar requisição #{} (algoritmo {}): {}", request_count, algorithm_name, e),
                }
            });
        }
    }
}
