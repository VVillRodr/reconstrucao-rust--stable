use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
// Propósito geral: fornecer os tipos de dados compartilhados 
// (mensagens entre cliente e servidor), e utilitários para representar 
// sucesso/erro nas operações de reconstrução.

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReconstructionResult {
    pub user_id: Uuid,
    pub algorithm_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub reconstruction_time_ms: i64,
    pub image_pixels: (usize, usize),
    pub iterations: usize,
    pub f: Array1<f64>,
}

// Adicionar uma implementação para facilitar a criação de erros.
impl ReconstructionResult {
    pub fn new_error(user_id: Uuid, algorithm_id: String) -> Self {
        let now = Utc::now();
        Self {
            user_id,
            algorithm_id,
            start_time: now,
            end_time: now,
            reconstruction_time_ms: 0,
            image_pixels: (0, 0),
            iterations: 0,
            f: Array1::zeros(0),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReconstructionRequest {
    pub user_id: Uuid, // ALTERADO: de u32 para Uuid
    pub model_id: String,
    pub algorithm_id: String,
    pub g: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ServerStatus {
    pub cpu_usage: f32,
    pub memory_usage_mb: u64,
    pub total_memory_mb: u64,
    pub queued_jobs: usize,
    pub active_jobs: usize,
    pub total_requests: u64,
    pub rejected_requests: u64,
    pub completed_requests: u64,
}

impl Default for ReconstructionResult {
    fn default() -> Self {
        Self {
            user_id: Uuid::nil(),
            algorithm_id: String::from("ERROR"),
            start_time: Utc::now(),
            end_time: Utc::now(),
            reconstruction_time_ms: 0,
            image_pixels: (0, 0),
            iterations: 0,
            f: Array1::zeros(0),
        }
    }
}

