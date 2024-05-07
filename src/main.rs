use anyhow::Error as E;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;

use tokenizers::Tokenizer;
//use tracing::info;

use futures::TryStreamExt;

use lancedb::*;
use lancedb::index::Index;
use lancedb::query::ExecutableQuery;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::arrow::arrow_array::{FixedSizeListArray, GenericByteArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use lancedb::arrow::arrow_array::types::{ByteArrayType, Float32Type};

const CLIP_EMBEDDING_DT: DataType = DataType::Float32;
type ClipEmbeddingType = Float32Type; // why are both of these necessary?
const CLIP_EMBEDDING_WIDTH: usize = 512;


// both the lancedb and the arrow-array APIs are really not great


use std::sync::Arc;

trait ToArc {
    fn to_arc(self) -> Arc<Self> where Self: Sized {
        Arc::new(self)
    }
}

impl ToArc for Schema {}
impl ToArc for Int32Array {}
impl<T: ByteArrayType> ToArc for GenericByteArray<T> {}
impl ToArc for FixedSizeListArray {}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();

    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    // .unsqueeze(0)?;
    Ok(img)
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];

    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?;

    Ok(images)
}

// NOTE: we're doing a lot of blocking here (downloading the model on first run, opening image
// files, and of course doing the embeddings) but we're only doing async stuff after that so it's
// fine.
#[tokio::main]
async fn main() -> anyhow::Result<()> {

    //tracing_subscriber::fmt::init();

    let api = hf_hub::api::sync::Api::new()?;

    let api = api.repo(hf_hub::Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/15".to_string(),
    ));


    let model_file = api.get("model.safetensors")?;
    let tokenizer = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    let config = clip::ClipConfig::vit_base_patch32();
    
    let device = Device::new_cuda(0)?;

    let vec_imgs = vec![
            "data/challah.jpg".to_string(),
            "data/banana_bread.jpg".to_string(),
            "data/robber.jpg".to_string(),
            "data/sauce.jpg".to_string(),
    ];

    let sequences = vec![
        "challah bread".into(),
        "banana bread".into(),
        "robber".into(),
        "marinara sauce".into(),
    ];

    // let image = load_image(args.image, config.image_size)?.to_device(&device)?;
    let images = load_images(&vec_imgs, config.image_size)?.to_device(&device)?;

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)? };

    let model = clip::ClipModel::new(vb, &config)?;

    let (input_ids, vec_seq) = tokenize_sequences(sequences, &tokenizer, &device)?;

    let image_features = model.get_image_features(&images)?;
    let text_features = model.get_text_features(&input_ids)?;

    let text_embeddings = div_l2_norm(&text_features)?;
    let image_embeddings = div_l2_norm(&image_features)?;

    println!("img {:?} text {:?}", image_embeddings, text_embeddings);

    // // manually compare
    // for i in 0..vec_seq.len() {
    //     let text = &vec_seq[i];
    //     let t_e = text_embeddings.get(i)?;
    //     let mut scores: Vec<f32> = vec![];
    //     for j in 0..vec_seq.len() {
    //         let i_e = image_embeddings.get(j)?;
    //         let score = compute_similarity(&t_e, &i_e)?;
    //         scores.push(score);
    //     }
    //     println!("text {}\nsim scores: {:?}", text, scores);
    // }

    // // embedding debug info
    // println!("img dim {:?}", image_embeddings.dims());
    // println!("txt dim {:?}", text_embeddings.dims());

    // for i in 0..text_embeddings.dims()[0] {
    //     println!("v{i}: {:?}", text_embeddings.get(i)
    //         .unwrap().to_vec1::<f32>().unwrap().len());
    // }
    
    let table = create_db("/tmp/testtable.ltable").await;
    insert_vectors(&table, &vec_seq, &image_embeddings).await;
    // can't create index with default parameters when giving only 4 vectors
    //create_index(&table).await;
    for i in 0..vec_seq.len() {
        let text = &vec_seq[i];
        let t_e = text_embeddings.get(i)?.to_vec1().unwrap();
        let scores = get_matches(&table, &t_e).await;
        println!("text {}\nscores: {:?}", text, scores);
    }

    Ok(())
}

// basically directly from the lancedb examples
fn db_schema() -> SchemaRef {
    Schema::new(vec![
        Field::new("file", DataType::Utf8, false),
        Field::new(
            "vector",
            // this has to be "item" in order to use from_iter_primitive
            DataType::FixedSizeList(Field::new("item", CLIP_EMBEDDING_DT, true).into(), CLIP_EMBEDDING_WIDTH as i32),
            false,
        ),
    ]).into()
}

async fn create_db(f: &str) -> lancedb::Table {
    let schema = db_schema();
    let db = lancedb::connection::connect(f)
        .execute()
        .await
        .unwrap();

    let table = db.create_empty_table("image_embeddings", schema)
        .execute()
        .await
        .unwrap();

    table
}

async fn insert_vectors(table: &Table, filenames: &[String], img_embeds: &Tensor) {
    let schema = db_schema();
    let mut vecs: Vec<_> = vec![];
    for i in 0..img_embeds.dims()[0] {
        let vec: Vec<_> = img_embeds
            .get(i)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(|f| Some(f)) // for from_iter_primitive
            .collect(); 
        vecs.push(Some(vec)); // again for from_iter_primitive
    }
    // StringArray::from is missing an impl for &[String]
    let filenames = filenames.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
    let files_array = StringArray::from(filenames); // why is the datatype utf8 but this is called
                                                    // stringarray

    let vectors_array = FixedSizeListArray::from_iter_primitive::<ClipEmbeddingType, _, _>(vecs, CLIP_EMBEDDING_WIDTH as i32);
    let batch = RecordBatch::try_new(schema.clone(),
        vec![files_array.to_arc(), vectors_array.to_arc()],
    ).unwrap();
    // this unwrap -> into iter -> map ok is dumb there should 100% be a better way to do this but
    // that's what the example code all has
    let records = RecordBatchIterator::new(vec![batch,].into_iter().map(Ok), schema);

    table.add(records)
        .execute()
        .await
        .unwrap();
}

async fn create_index(table: &Table) {
	table.create_index(&["vector"], Index::Auto)
	   .execute()
	   .await
	   .unwrap();
}

async fn get_matches(table: &Table, text_vec: &[f32]) -> Vec<String> {
    let matches = table
        .vector_search(text_vec)
        .unwrap()
        .execute()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    // I'm not sure when or if this can return more than 1 batch
    let batch = &matches[0];
    let output: Vec<String> = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Failed to downcast")
        .iter()
        .map(|o| o.unwrap().into())
        .collect();
    output
}
fn compute_similarity(v1: &Tensor, v2: &Tensor) -> anyhow::Result<f32> {
    let sum_12 = (v1 * v2)?.sum_all()?.to_scalar::<f32>()?;
    let sum_11 = (v1 * v1)?.sum_all()?.to_scalar::<f32>()?;
    let sum_22 = (v2 * v2)?.sum_all()?.to_scalar::<f32>()?;
    Ok(sum_12 / (sum_11 * sum_22).sqrt())
}

pub fn div_l2_norm(v: &Tensor) -> anyhow::Result<Tensor> {
    let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(v.broadcast_div(&l2_norm)?)
}


pub fn tokenize_sequences(
    sequences: Vec<String>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(E::msg("No pad token"))?;

    let vec_seq = sequences;

    let mut tokens = vec![];

    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }

    let input_ids = Tensor::new(tokens, device)?;

    Ok((input_ids, vec_seq))
}
