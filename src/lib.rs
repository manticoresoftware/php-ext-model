#![cfg_attr(windows, feature(abi_vectorcall))]
use ext_php_rs::prelude::*;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json::Value;
use tokenizers::Tokenizer;

fn build_model_and_tokenizer(
    model_id: String,
    revision: String,
    use_pth: bool,
) -> Result<(BertModel, Tokenizer, usize, usize)> {
    let device = Device::Cpu;
    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = if use_pth {
            api.get("pytorch_model.bin")?
        } else {
            api.get("model.safetensors")?
        };
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let max_input_len = get_max_input_length(&config)?;
    let hidden_size = get_hidden_size(&config)?;
    let mut config: Config = serde_json::from_str(&config)?;
    let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = if use_pth {
        VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
    };
    config.hidden_act = HiddenAct::GeluApproximate;
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer, max_input_len, hidden_size))
}

/// Get maximum input length for sequence for the current model
fn get_max_input_length(contents: &str) -> Result<usize> {
    let config: Value = serde_json::from_str(&contents)?;
    let max_length = config["max_position_embeddings"]
        .as_u64()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Max position embeddings not found"))?;
    Ok(max_length as usize)
}

fn get_hidden_size(contents: &str) -> Result<usize> {
    let config: Value = serde_json::from_str(&contents)?;
    let max_length = config["hidden_size"]
        .as_u64()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Hidden size not found"))?;
    Ok(max_length as usize)
}

#[php_class(name = "Manticore\\Ext\\Model")]
struct Model {
    model: BertModel,
    tokenizer: Tokenizer,
    max_input_len: usize,
    hidden_size: usize,
}

#[php_impl(rename_methods = "camelCase")]
impl Model {
    /// Static method to instantiate the Tokenizer
    #[php_static_method]
    /// @param string $model_id name of the model to use from the huggingface.co
    /// @param ?string $revision The revision of the mode to use, default is string
    /// @param bool $use_pth If we use pytorch model or safetensors
    /// @return self Instance of created class
    pub fn create(model_id: String, revision: Option<String>, use_pth: Option<bool>) -> Self {
        let revision = revision.unwrap_or("main".to_string());
        let use_pth = use_pth.unwrap_or(false);
        let (model, tokenizer, max_input_len, hidden_size) =
            build_model_and_tokenizer(model_id, revision, use_pth).unwrap();
        Model {
            model,
            tokenizer,
            max_input_len,
            hidden_size,
        }
    }

    /// Get maximum input len in tokens allowed for this model
    /// @return int
    #[php]
    pub fn get_max_input_len(&mut self) -> usize {
        self.max_input_len
    }

    /// Get resulting vector size for embeddings for this model
    /// @return int
    pub fn get_hidde_size(&mut self) -> usize {
        self.hidden_size
    }

    /// @param string $text Text to convert into the token and return array of it
    /// @return array<string>
    pub fn predict(&mut self, text: String) -> Vec<f32> {
        let device = &self.model.device;
        let tokenizer = self
            .tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)
            .unwrap();
        let tokens = tokenizer
            .encode(text.clone(), true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();

        let chunks = chunk_input_tokens(&tokens, self.max_input_len, (self.max_input_len / 10) as usize);
        let mut results: Vec<Vec<f32>> = Vec::new();
        for chunk in &chunks {
            let token_ids = Tensor::new(&chunk[..], device).unwrap().unsqueeze(0).unwrap();
            let token_type_ids = token_ids.zeros_like().unwrap();
            let embeddings = self.model.forward(&token_ids, &token_type_ids).unwrap();

            // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
            let (n_sentences, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
            let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();

            for j in 0..n_sentences {
                let e_j = embeddings.get(j).unwrap();
                let mut emb: Vec<f32> = e_j.to_vec1().unwrap();
                normalize(&mut emb);
                results.push(emb);
                break;
            }
        }
        get_mean_vector(&results)
    }
}

/// Module initialization
#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
}

pub fn normalize(v: &mut Vec<f32>) {
    let length: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= length);
}

fn chunk_input_tokens(tokens: &[u32], max_seq_len: usize, stride: usize) -> Vec<Vec<u32>> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < tokens.len() {
        let end = std::cmp::min(start + max_seq_len, tokens.len());
        let chunk = tokens[start..end].to_vec();
        chunks.push(chunk);
        start += max_seq_len - stride;
    }

    chunks
}

fn get_mean_vector(results: &Vec<Vec<f32>>) -> Vec<f32> {
    if results.is_empty() {
        return Vec::new();
    }

    let num_cols = results[0].len();
    let mut mean_vector = vec![0.0; num_cols];

    let mut weight_sum = 0.0;

    for (i, row) in results.iter().enumerate() {
        let weight = if i == 0 { 1.2 } else { 1.0 }; // Adjust the weight for the first chunk here
        weight_sum += weight;

        for (j, val) in row.iter().enumerate() {
            mean_vector[j] += weight * val;
        }
    }

    for val in &mut mean_vector {
        *val /= weight_sum;
    }

    mean_vector
}
