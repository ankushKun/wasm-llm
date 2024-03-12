use llm::{InferenceFeedback, KnownModel};
use std::{convert::Infallible, path::Path};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn test() -> String {
    let str = "test OK!";

    str.to_string()
}

#[wasm_bindgen]
pub struct Myllm {
    model: llm::models::Bloom,
    session: llm::InferenceSession,
}

#[wasm_bindgen]
impl Myllm {
    pub fn new() -> Myllm {
        let m = llm::load::<llm::models::Bloom>(
            Path::new("./models/bloom-3b-q5_1.bin"),
            llm::TokenizerSource::Embedded,
            Default::default(),
            |progress| {
                println!("{progress:?}");
            },
        )
        .unwrap();
        let s = m.start_session(Default::default());
        Myllm {
            model: m,
            session: s,
        }
    }

    pub fn infer(&mut self, input: &str) -> String {
        println!("Running inference for input: {}", input);

        let infer_res = self.session.infer::<std::convert::Infallible>(
            // model to use for text generation
            &self.model,
            // randomness provider
            &mut rand::thread_rng(),
            // the prompt to use for text generation, as well as other
            // inference parameters
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text("The sun is a deadly "),
                parameters: &Default::default(),
                play_back_previous_tokens: true,
                maximum_token_count: Some(100),
            },
            &mut Default::default(),
            // output callback
            |t| match t {
                llm::InferenceResponse::SnapshotToken(x) => {
                    print!("{}", x);
                    Result::<InferenceFeedback, Infallible>::Ok(InferenceFeedback::Continue)
                }
                llm::InferenceResponse::InferredToken(x) => {
                    print!("{}", x);
                    Result::<InferenceFeedback, Infallible>::Ok(InferenceFeedback::Continue)
                }
                llm::InferenceResponse::EotToken => {
                    Result::<InferenceFeedback, Infallible>::Ok(InferenceFeedback::Continue)
                }
                llm::InferenceResponse::PromptToken(x) => {
                    print!("{}", x);
                    Result::<InferenceFeedback, Infallible>::Ok(InferenceFeedback::Continue)
                }
            },
        );

        match infer_res {
            Ok(result) => {
                println!("\n\nInference stats:\n{result}");
                // print inference text
                println!("{}", self.session.decoded_tokens()[0]);
                let tokens = self.session.decoded_tokens();
                let mut res = String::new();
                for i in tokens {
                    res.push_str(i.to_string().as_str());
                }
                String::from(res)
            }
            Err(err) => {
                println!("\n{err}");
                String::from("Error")
            }
        }
    }
}
