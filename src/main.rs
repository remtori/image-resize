use std::{
    net::SocketAddr,
    num::NonZeroU32,
    path::PathBuf,
    time::{Duration, Instant},
};

use axum::{
    body,
    error_handling::HandleErrorLayer,
    extract::{Extension, Path, Query},
    http::{self, header, Method, StatusCode},
    response::{AppendHeaders, IntoResponse},
    routing::get,
    BoxError, Router,
};
use clap::Parser;
use fast_image_resize as fir;
use image::{codecs::jpeg::JpegEncoder, ImageEncoder};
use reqwest::Client;
use serde::Deserialize;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};

#[derive(Parser, Clone)]
#[clap(version)]
struct Cli {
    #[clap(short, long, value_parser)]
    port: Option<u16>,
    #[clap(short, long, value_parser)]
    remote_cdn: Option<String>,
    #[clap(short, long, value_parser)]
    local_folder: Option<String>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.remote_cdn.is_none() && cli.local_folder.is_none() {
        tracing::error!("Either 'remote_cdn' or 'local_folder' is required");
        return;
    }

    let client = Client::builder()
        .gzip(true)
        .brotli(true)
        .connect_timeout(Duration::new(1, 0))
        .build()
        .unwrap();

    let app = Router::new()
        .route("/*path", get(handler))
        .layer(Extension(client))
        .layer(Extension(cli.clone()))
        .layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::predicate(
                    |origin: &hyper::http::HeaderValue, _request_parts: &http::request::Parts| {
                        origin.as_bytes().ends_with(b".remtori.com")
                    },
                ))
                .allow_methods([Method::GET]),
        )
        .layer(
            tower::ServiceBuilder::new()
                .layer(HandleErrorLayer::new(handle_error))
                .timeout(Duration::from_secs(30)),
        )
        .layer(TraceLayer::new_for_http());

    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port.unwrap_or(3000)));

    tracing::info!("Running image resize server with:");
    if let Some(folder) = cli.local_folder {
        tracing::info!("\tlocal folder: {folder}");
    }
    if let Some(url) = cli.remote_cdn {
        tracing::info!("\tremote cdn: {url}");
    }

    tracing::info!("Listening on {}", addr);
    axum::Server::bind(&addr).serve(app.into_make_service()).await.unwrap();
}

async fn handle_error(err: BoxError) -> StatusCode {
    tracing::warn!("Unhandled error: {err:#}");
    StatusCode::NOT_FOUND
}

async fn handler(
    Extension(client): Extension<Client>,
    Extension(config): Extension<Cli>,
    Query(params): Query<Params>,
    Path(path): Path<String>,
) -> impl IntoResponse {
    let start = Instant::now();

    let mut bytes = None;
    if let Some(file_path) = config.local_folder {
        let mut file_path = PathBuf::from(file_path);
        file_path.push(&path);

        match tokio::fs::read(file_path).await {
            Ok(data) => bytes = Some(bytes::Bytes::from(data)),
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    tracing::error!(path, "Read local file error {err:#}");
                }
            }
        }
    }

    if bytes.is_none() {
        if let Some(mut url) = config.remote_cdn {
            if url.chars().last().unwrap() == '/' {
                url.push_str(&path[1..]);
            } else {
                url.push_str(&path);
            }

            match client.get(url).send().await {
                Ok(resp) if resp.status().is_success() => match resp.bytes().await {
                    Ok(data) => bytes = Some(data),
                    Err(err) => {
                        tracing::error!(path, "Request get bytes error {err:#}");
                    }
                },
                Ok(resp) => {
                    tracing::info!(path, "Request error: status code {}", resp.status());
                }
                Err(err) => {
                    tracing::info!(path, "Request error {err:#}");
                }
            }
        }
    }

    let time_fetch = start.elapsed();
    let start = Instant::now();
    let bytes = bytes.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            AppendHeaders([(header::CACHE_CONTROL, "public, s-max-age=28800")]),
        )
            .into_response()
    })?;

    let image = image::load_from_memory(&bytes[..]).map_err(|err| {
        // Cache this response since this file most likely is not an image
        tracing::error!(path, "Decode image error {err:#}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            AppendHeaders([(header::CACHE_CONTROL, "public, s-max-age=604800")]),
            "Decode image error",
        )
            .into_response()
    })?;

    let time_decode = start.elapsed();
    let start = Instant::now();
    let src_image = fir::Image::from_vec_u8(
        NonZeroU32::new(image.width()).unwrap(),
        NonZeroU32::new(image.height()).unwrap(),
        image.to_rgb8().into_raw(),
        fir::PixelType::U8x3,
    )
    .unwrap();

    let (dst_width, dst_height) = {
        let width = params.width.or(params.w);
        let height = params.height.or(params.h);

        if let (Some(width), Some(height)) = (width, height) {
            (width.get(), height.get())
        } else if let Some(width) = width {
            let ratio = width.get() as f32 / src_image.width().get() as f32;
            (width.get(), (src_image.height().get() as f32 * ratio) as u32)
        } else if let Some(height) = height {
            let ratio = height.get() as f32 / src_image.height().get() as f32;
            ((src_image.width().get() as f32 * ratio) as u32, height.get())
        } else {
            (src_image.width().get() / 4, src_image.height().get() / 4)
        }
    };

    let mut dst_image = fir::Image::new(
        NonZeroU32::new(dst_width).unwrap(),
        NonZeroU32::new(dst_height).unwrap(),
        src_image.pixel_type(),
    );

    let mut resizer = fir::Resizer::new(fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3));
    if let Err(err) = resizer.resize(&src_image.view(), &mut dst_image.view_mut()) {
        tracing::error!(path, "Resize image error {err:#}");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            AppendHeaders([(header::CACHE_CONTROL, "public, s-max-age=28800")]),
            "Resize image error",
        )
            .into_response());
    }

    let time_resize = start.elapsed();
    let start = Instant::now();

    let mut result_buf = Vec::new();
    JpegEncoder::new(&mut result_buf)
        .write_image(
            dst_image.buffer(),
            dst_image.width().get(),
            dst_image.height().get(),
            image::ColorType::Rgb8,
        )
        .unwrap();

    let time_encode = start.elapsed();
    tracing::info!(
        "Image processed path={} original={}x{} resized={}x{} fetch={}ms decode={}ms resize={}ms encode={}ms",
        path,
        src_image.width(),
        src_image.height(),
        dst_image.width(),
        dst_image.height(),
        time_fetch.as_millis(),
        time_decode.as_millis(),
        time_resize.as_millis(),
        time_encode.as_millis(),
    );

    Ok((
        AppendHeaders([
            (header::CONTENT_TYPE, "image/jpeg"),
            (header::CACHE_CONTROL, "public, s-max-age=2592000"),
        ]),
        body::Full::new(bytes::Bytes::from(result_buf)),
    ))
}

#[derive(Debug, Deserialize)]
struct Params {
    width: Option<NonZeroU32>,
    height: Option<NonZeroU32>,
    w: Option<NonZeroU32>,
    h: Option<NonZeroU32>,
}
