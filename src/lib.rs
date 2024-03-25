#![cfg_attr(windows, feature(abi_vectorcall))]
use ext_php_rs::prelude::*;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_transformers::models::bert::{BertModel, HiddenAct, Config, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn build_model_and_tokenizer(model_id: String, revision: String, use_pth: bool) -> Result<(BertModel, Tokenizer)> {
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
  let mut config: Config = serde_json::from_str(&config)?;
  let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_filename)
		.map_err(E::msg)?;

  let vb = if use_pth {
    VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
  } else {
    unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
  };
  config.hidden_act = HiddenAct::GeluApproximate;
  let model = BertModel::load(vb, &config)?;
  Ok((model, tokenizer))
}

#[php_class(name = "Manticore\\Ext\\Model")]
struct Model {
	model: BertModel,
	tokenizer: Tokenizer,
}

#[php_impl]
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
  	let (model, tokenizer) = build_model_and_tokenizer(model_id, revision, use_pth).unwrap();
    Model {
    	model,
    	tokenizer
    }
  }

  /// @param string $text Text to convert into the token and return array of it
  /// @return array<string>
	pub fn predict(&mut self, text: String) -> Vec<f32> {
		let device = &self.model.device;
  	let tokenizer = self.tokenizer
	    .with_padding(None)
	    .with_truncation(None)
	    .map_err(E::msg).unwrap();
		let tokens = tokenizer
			.encode(text.clone(), true)
			.map_err(E::msg).unwrap()
			.get_ids()
			.to_vec();
		let token_ids = Tensor::new(&tokens[..], device).unwrap().unsqueeze(0).unwrap();
		let token_type_ids = token_ids.zeros_like().unwrap();
		let ys = self.model.forward(&token_ids, &token_type_ids).unwrap();
		let cls_embedding = ys.get(0).unwrap().get(0).unwrap();
		let ys_vec = cls_embedding.flatten_all().unwrap().to_vec1().unwrap();
		ys_vec
	}
}


/// Module initialization
#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
	module
}
