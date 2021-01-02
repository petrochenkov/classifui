#![feature(array_windows)]
#![feature(format_args_capture)]
#![feature(total_cmp)]

use clap::App;
use fxhash::FxHashMap as HashMap;
use walkdir::WalkDir;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::error::Error;
use std::path::Path;
use std::{fs, mem};

pub fn is_id_continue(c: char) -> bool {
    // This is exactly XID_Continue.
    // We also add fast-path for ascii idents
    ('a'..='z').contains(&c)
        || ('A'..='Z').contains(&c)
        || ('0'..='9').contains(&c)
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_continue(c))
}

enum State {
    Start,
    ContinueIdent,
    ContinuePunct,
}

fn tokenize(s: &str) -> Vec<String> {
    let mut state = State::Start;
    let mut res = Vec::new();
    let mut curtok = String::new();
    for c in s.chars() {
        if c.is_whitespace() {
            if !curtok.is_empty() {
                res.push(mem::take(&mut curtok));
            }
            state = State::Start;
        } else if is_id_continue(c) {
            match state {
                State::Start | State::ContinueIdent => {}
                State::ContinuePunct => {
                    assert!(!curtok.is_empty());
                    res.push(mem::take(&mut curtok));
                }
            }
            curtok.push(c);
            state = State::ContinueIdent;
        } else {
            // Punct
            match state {
                State::Start | State::ContinuePunct => {}
                State::ContinueIdent => {
                    assert!(!curtok.is_empty());
                    res.push(mem::take(&mut curtok));
                }
            }
            curtok.push(c);
            state = State::ContinuePunct;
        }
    }

    if !curtok.is_empty() {
        res.push(mem::take(&mut curtok));
    }

    res
}

fn tokens_to_features(
    feature_map: &mut FeatureMap,
    tokens: &[String],
    read_only: bool,
) -> Vec<u32> {
    let mut res = Vec::new();
    for token in tokens {
        if let Some(feat) = feature_map.intern(token.into(), read_only) {
            res.push(feat);
        }
    }
    for [token1, token2] in tokens.array_windows() {
        if let Some(feat) = feature_map.intern(format!("{} {}", token1, token2).into(), read_only) {
            res.push(feat);
        }
    }
    for [token1, _, token3] in tokens.array_windows() {
        if let Some(feat) = feature_map.intern(format!("{}  {}", token1, token3).into(), read_only)
        {
            res.push(feat);
        }
    }
    for [token1, token2, token3] in tokens.array_windows() {
        if let Some(feat) =
            feature_map.intern(format!("{} {} {}", token1, token2, token3).into(), read_only)
        {
            res.push(feat);
        }
    }
    res.sort_unstable();
    res.dedup();
    res
}

#[derive(Default)]
struct FeatureMap {
    features: Vec<String>,
    map: HashMap<String, u32>,
}

impl FeatureMap {
    fn intern(&mut self, feature: Cow<str>, read_only: bool) -> Option<u32> {
        if let Some(index) = self.map.get(&*feature) {
            Some(*index)
        } else if read_only {
            None
        } else {
            let new_index = u32::try_from(self.features.len()).unwrap();
            self.features.push(feature.clone().into_owned());
            self.map.insert(feature.into_owned(), new_index);
            Some(new_index)
        }
    }
}

fn train() -> Result<(), Box<dyn Error>> {
    let root = Path::new("C:/msys64/home/we/rust/src/test/ui");
    let mut feature_map = FeatureMap::default();
    feature_map.features.push(String::new()); // feature indices must start with 1
    let mut classes = BTreeMap::<String, Vec<Vec<u32>>>::default();
    for top_entry in fs::read_dir(root)? {
        let top_entry = top_entry?;
        if !top_entry.file_type()?.is_dir()
            || top_entry.file_name() == "auxiliary"
            || top_entry.file_name() == "issues"
        {
            continue;
        }

        let top_path = top_entry.path();
        let class = top_path.file_name().unwrap().to_str().unwrap();
        for entry in
            WalkDir::new(&top_path).into_iter().filter_entry(|e| e.file_name() != "auxiliary")
        {
            let entry = entry?;
            if !entry.file_type().is_dir() {
                let path = entry.path();
                if let Ok(s) = fs::read_to_string(path) {
                    classes.entry(class.to_owned()).or_default().push(tokens_to_features(
                        &mut feature_map,
                        &tokenize(&s),
                        false,
                    ));
                }
            }
        }
    }

    // fs::write("features.txt", &format!("{:#?}", res))?;
    fs::create_dir_all("model/classes").ok();
    let feature_map_str = serde_json::to_string(&feature_map.map)?;
    fs::write("model/features.json", feature_map_str)?;

    for (class_to_train, _) in &classes {
        println!("Training model for {}", class_to_train);
        let mut labels = Vec::new();
        let mut features = Vec::new();
        for (class, vectors) in &classes {
            let label = if class == class_to_train { 1.0 } else { 0.0 };
            for vector in vectors {
                labels.push(label);
                features.push(vector.iter().copied().map(|i| (i, 1.0)).collect());
            }
        }

        use liblinear::LibLinearModel;
        use liblinear::SolverType;

        let input_data = liblinear::util::TrainingInput::from_sparse_features(labels, features)
            .map_err(|e| e.to_string())?;

        let mut builder = liblinear::Builder::new();
        builder.problem().input_data(input_data);
        builder.parameters().solver_type(SolverType::L1R_L2LOSS_SVC);
        let model = builder.build_model()?;
        model
            .save_to_disk(&format!("model/classes/{}.txt", class_to_train))
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn classify() -> Result<(), Box<dyn Error>> {
    let root = Path::new("C:/msys64/home/we/rust/src/test/ui");
    let feature_map_str = fs::read_to_string("model/features.json")?;
    let feature_map: HashMap<String, u32> = serde_json::from_str(&feature_map_str)?;
    let mut feature_map = FeatureMap { map: feature_map, features: Vec::new() };

    use liblinear::util::PredictionInput;
    use liblinear::LibLinearModel;
    use liblinear::Serializer;

    let mut models = HashMap::default();
    for entry in fs::read_dir("model/classes")? {
        let entry = entry?;
        println!("Loading model: {:?}", entry);
        let model_path = entry.path();
        let model_name =
            model_path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_owned()).unwrap();
        let model = Serializer::load_model(model_path.to_str().unwrap())?;
        models.insert(model_name, model);
    }

    for dir in &[&root.join("issues"), root] {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                let path = entry.path();
                let test_name =
                    path.file_name().and_then(|s| s.to_str()).map(|s| s.to_owned()).unwrap();
                if let Ok(s) = fs::read_to_string(&path) {
                    let features = tokens_to_features(&mut feature_map, &tokenize(&s), true);
                    let mut model_scores = Vec::new();
                    for (model_name, model) in &models {
                        let mut dense_features = vec![0.0; feature_map.map.len()];
                        for &feat in &features {
                            dense_features[feat as usize - 1] = 1.0;
                        }

                        // FIXME: `from_sparse_features` in liblinear-rs doesn't work correctly.
                        let prediction_input = PredictionInput::from_dense_features(dense_features)
                            .map_err(|e| e.to_string())?;
                        let predict =
                            model.predict_values(prediction_input).map_err(|e| e.to_string())?;
                        let score = predict.0[0];
                        let score = if model.labels()[0] == 1 { score } else { -score };
                        model_scores.push((model_name, score));
                    }

                    model_scores.sort_by(|(_, sc1), (_, sc2)| sc1.total_cmp(&sc2));
                    let mut msg = format!("{test_name}: ");
                    for (i, (name, score)) in model_scores.iter().rev().take(3).enumerate() {
                        if i != 0 {
                            msg.push_str(", ");
                        }
                        msg.push_str(&format!("{name} ({score:.3})"));
                    }
                    println!("{}", msg);
                }
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = App::new("UI test classifier")
        .args_from_usage(
            "--train 'Train the classifier'
                               --classify 'Classify tests'",
        )
        .get_matches();

    if matches.is_present("train") {
        train()?;
    }
    if matches.is_present("classify") {
        classify()?;
    }

    Ok(())
}
