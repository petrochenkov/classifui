#![feature(array_windows)]
#![feature(format_args_capture)]
#![feature(total_cmp)]

use clap::App;
use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use liblinear::util::TrainingInput;
use liblinear::{Builder as LiblinearBuilder, LibLinearModel as _, SolverType};
use regex::Regex;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use std::borrow::Cow;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::error::Error;
use std::path::Path;
use std::{fs, mem};

enum LexerState {
    Start,
    ContinueIdent,
    ContinuePunct,
}

/// Mapping string-based features to integer indices and back.
#[derive(Default)]
struct FeatureMap {
    features: Vec<String>,
    map: HashMap<String, u32>,
}

/// Linear SVM model produced by training and used during classification.
#[derive(Serialize, Deserialize)]
struct Model {
    features: HashMap<String, u32>,
    classes: HashMap<String, Vec<(u32, f64)>>,
}

struct ClassifiedTest {
    name: String,
    class_scores: Vec<(String, f64)>,
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

impl ClassifiedTest {
    fn max_score(&self) -> f64 {
        self.class_scores[0].1
    }
}

fn is_id_start(c: char) -> bool {
    // This is XID_Start OR '_' (which formally is not a XID_Start).
    // We also add fast-path for ascii idents
    ('a'..='z').contains(&c)
        || ('A'..='Z').contains(&c)
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_start(c))
}

fn is_id_continue(c: char) -> bool {
    // This is exactly XID_Continue.
    // We also add fast-path for ascii idents
    ('a'..='z').contains(&c)
        || ('A'..='Z').contains(&c)
        || ('0'..='9').contains(&c)
        || c == '_'
        || (c > '\x7f' && unicode_xid::UnicodeXID::is_xid_continue(c))
}

/// Turn text of a test into tokens.
fn tokenize(s: &str) -> Vec<String> {
    let mut state = LexerState::Start;
    let mut res = Vec::new();
    let mut curtok = String::new();
    for c in s.chars() {
        if c.is_whitespace() {
            if !curtok.is_empty() {
                res.push(mem::take(&mut curtok));
            }
            state = LexerState::Start;
        } else if is_id_continue(c) {
            match state {
                LexerState::Start | LexerState::ContinueIdent => {}
                LexerState::ContinuePunct => {
                    assert!(!curtok.is_empty());
                    res.push(mem::take(&mut curtok));
                }
            }
            curtok.push(c);
            state = LexerState::ContinueIdent;
        } else {
            // Punct
            match state {
                LexerState::Start | LexerState::ContinuePunct => {}
                LexerState::ContinueIdent => {
                    assert!(!curtok.is_empty());
                    res.push(mem::take(&mut curtok));
                }
            }
            curtok.push(c);
            state = LexerState::ContinuePunct;
        }
    }

    if !curtok.is_empty() {
        res.push(mem::take(&mut curtok));
    }

    res
}

/// Turns all identifiers and digits into a single token.
fn generalize(s: &str) -> &str {
    let first_char = s.chars().next().unwrap();
    if is_id_continue(first_char) {
        if is_id_start(first_char) { "и" } else { "ц" }
    } else {
        s
    }
}

/// Turn tokens of a test into features (in their index representation).
/// Tokens, "generalized" tokens, and their bigrams and trigrams are used as features.
fn tokens_to_features(
    feature_map: &mut FeatureMap,
    tokens: &[String],
    read_only: bool,
) -> Vec<u32> {
    let mut res = Vec::new();
    let mut push = |token| {
        if let Some(feat) = feature_map.intern(token, read_only) {
            res.push(feat);
        }
    };
    for token in tokens {
        push(token.into());
        push(generalize(token).into());
    }
    for [token1, token2] in tokens.array_windows() {
        push(format!("{} {}", token1, token2).into());
        push(format!("{} {}", generalize(token1), generalize(token2)).into());
    }
    for [token1, _, token3] in tokens.array_windows() {
        push(format!("{}  {}", token1, token3).into());
        push(format!("{}  {}", generalize(token1), generalize(token3)).into());
    }
    for [token1, token2, token3] in tokens.array_windows() {
        push(format!("{} {} {}", token1, token2, token3).into());
        push(
            format!("{} {} {}", generalize(token1), generalize(token2), generalize(token3)).into(),
        );
    }
    res.sort_unstable();
    res.dedup();
    res
}

/// Merge features from `foo.rs` and `foo.stderr` into a single feature vector
/// that corresponds to a single test case including multiple files.
fn files_to_tests(files: HashMap<String, RefCell<Vec<u32>>>) -> HashMap<String, Vec<u32>> {
    let mut res = HashMap::default();
    for (name, features) in &files {
        let mut key = name.to_string();
        let prefix = if let prefix @ Some(_) = name.strip_suffix(".nll.stderr") {
            prefix
        } else if let prefix @ Some(_) = name.strip_suffix(".stderr") {
            prefix
        } else if let prefix @ Some(_) = name.strip_suffix(".stdout") {
            prefix
        } else if let prefix @ Some(_) = name.strip_suffix(".fixed") {
            prefix
        } else {
            None
        };
        if let Some(prefix) = prefix {
            let normalized = prefix.to_string() + ".rs";
            if files.contains_key(&normalized) {
                key = normalized;
            }
        }

        merge_features(res.entry(key).or_default(), &mut features.borrow_mut());
    }
    res
}

fn merge_features(dst: &mut Vec<u32>, src: &mut Vec<u32>) {
    dst.append(src);
    dst.sort_unstable();
    dst.dedup();
}

/// Dot product of weight vector from the trained linear model
/// and feature vector from a new test case that needs to be classified.
/// Both vectors are sparse.
fn get_decision_value(m: &[(u32, f64)], x: &[u32]) -> f64 {
    let mut res = 0.0;
    for index in x {
        match m.binary_search_by_key(index, |node| node.0) {
            Ok(i) => res += m[i].1,
            Err(..) => {}
        }
    }
    res
}

/// Train classifier and write it to `model.json`.
fn train(root: &Path) -> Result<(), Box<dyn Error>> {
    // Build feature vectors for already classified tests.
    let mut feature_map = FeatureMap::default();
    feature_map.features.push(String::new()); // feature indices must start with 1
    let mut class_vectors = Vec::new();
    for top_entry in fs::read_dir(root)? {
        let top_entry = top_entry?;
        if !top_entry.file_type()?.is_dir()
            || top_entry.file_name() == "auxiliary"
            || top_entry.file_name() == "issues"
            || top_entry.file_name() == "error-codes"
            || top_entry.file_name() == "rfcs"
        {
            continue;
        }

        let top_path = top_entry.path();
        let class = top_path.file_name().unwrap().to_str().unwrap();
        let mut files = HashMap::default();
        for entry in
            WalkDir::new(&top_path).into_iter().filter_entry(|e| e.file_name() != "auxiliary")
        {
            let entry = entry?;
            if !entry.file_type().is_dir() {
                let path = entry.path();
                if let Ok(s) = fs::read_to_string(path) {
                    let file_name =
                        path.strip_prefix(root)?.display().to_string().replace("\\", "/");
                    let features = tokens_to_features(&mut feature_map, &tokenize(&s), false);
                    files.insert(file_name, RefCell::new(features));
                }
            }
        }

        class_vectors.push((class.to_owned(), files_to_tests(files)));
    }

    // Turn feature vectors into input for liblinear.
    let mut labels = Vec::new();
    let mut features = Vec::new();
    for (class_idx, (_, vectors)) in class_vectors.iter().enumerate() {
        for (_, vector) in vectors {
            labels.push(class_idx as f64);
            features.push(vector.iter().copied().map(|i| (i, 1.0)).collect());
        }
    }
    let input_data =
        TrainingInput::from_sparse_features(labels, features).map_err(|e| e.to_string())?;

    // Train liblinear model.
    let mut builder = LiblinearBuilder::new();
    builder.problem().input_data(input_data);
    builder.parameters().solver_type(SolverType::L1R_L2LOSS_SVC);
    let liblinear_model = builder.build_model()?;

    // Convert the trained model into sparse representation.
    let mut classes = HashMap::default();
    let mut used_features = HashSet::default();
    for (class_idx, (class_name, _)) in class_vectors.iter().enumerate() {
        let class_idx = i32::try_from(class_idx).unwrap();
        let mut weights = Vec::new();
        for feature_index in 1..i32::try_from(liblinear_model.num_features()).unwrap() + 1 {
            let weight = liblinear_model.feature_coefficient(feature_index, class_idx);
            if weight != 0.0 {
                let index = u32::try_from(feature_index).unwrap();
                weights.push((index, weight));
                used_features.insert(index);
            }
        }

        classes.insert(class_name.clone(), weights);
    }

    // Throw away features that ended up unused from the table.
    let features =
        feature_map.map.into_iter().filter(|(_, index)| used_features.contains(index)).collect();

    // Write the model into file.
    // FIXME: Make the output model file configurable.
    let model = Model { features, classes };
    let model_str = serde_json::to_string(&model)?;
    fs::write("model.json", model_str)?;

    Ok(())
}

/// Read classifier from `model.json` and use it to classify tests.
fn classify(root: &Path) -> Result<(), Box<dyn Error>> {
    // Read the model from file.
    // FIXME: Make the input model file configurable.
    let model_str = fs::read_to_string("model.json")?;
    let mut model: Model = serde_json::from_str(&model_str)?;
    let mut feature_map = FeatureMap { map: mem::take(&mut model.features), features: Vec::new() };

    // Classify tests that are not yet classified using the model.
    let mut files = HashMap::default();
    for dir in &[&root.join("issues"), root] {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() && entry.file_name() != ".gitattributes" {
                let path = entry.path();
                if let Ok(s) = fs::read_to_string(&path) {
                    let file_name =
                        path.strip_prefix(root)?.display().to_string().replace("\\", "/");
                    let features = tokens_to_features(&mut feature_map, &tokenize(&s), true);
                    files.insert(file_name, RefCell::new(features));
                }
            }
        }
    }

    let mut classified_tests = Vec::new();
    for (name, features) in files_to_tests(files) {
        let mut model_scores = Vec::new();
        for (model_name, weights) in &model.classes {
            let score = get_decision_value(weights, &features);
            model_scores.push((model_name, score));
        }

        // Print three classes with highest decision values.
        model_scores.sort_by(|(_, sc1), (_, sc2)| sc1.total_cmp(&sc2));
        classified_tests.push(ClassifiedTest {
            name,
            class_scores: model_scores
                .into_iter()
                .rev()
                .take(3)
                .map(|(name, score)| (name.clone(), score))
                .collect(),
        });
    }

    let re = Regex::new(r"issue-(\d+)").unwrap();
    classified_tests.sort_by(|test1, test2| test2.max_score().total_cmp(&test1.max_score()));
    for test in classified_tests {
        let mut msg = format!(
            "- [{}](https://github.com/rust-lang/rust/blob/master/src/test/ui/{})",
            test.name, test.name
        );
        if let Some(captures) = re.captures(&test.name) {
            msg.push_str(&format!(
                " <sup>[issue](https://github.com/rust-lang/rust/issues/{})</sup>",
                &captures[1]
            ));
        }
        msg.push_str(": ");
        for (i, (name, score)) in test.class_scores.iter().enumerate() {
            if i != 0 {
                msg.push_str(", ");
            }
            msg.push_str(&format!("{name} ({score:.3})"));
        }
        println!("{}", msg);
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

    // FIXME: Make it configurable.
    let root = Path::new("C:/msys64/home/we/rust/src/test/ui");

    if matches.is_present("train") {
        train(root)?;
    }
    if matches.is_present("classify") {
        classify(root)?;
    }

    Ok(())
}
