use std::{
    net::SocketAddr,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use axum::{
    body,
    extract::{Extension, Path, Query},
    http::{self, header, Method, Request, StatusCode},
    response::{AppendHeaders, IntoResponse},
    routing::get,
    Router,
};
use fast_image_resize as fir;
use hyper::Body;
use image::{codecs::jpeg::JpegEncoder, ImageEncoder};
use reqwest::Client;
use serde::Deserialize;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let client = Client::builder()
        .gzip(true)
        .brotli(true)
        .connect_timeout(Duration::new(1, 0))
        .build()
        .unwrap();

    let app = Router::new()
        .route("/*path", get(handler))
        .layer(Extension(client))
        .layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::predicate(
                    |origin: &hyper::http::HeaderValue, _request_parts: &http::request::Parts| {
                        origin.as_bytes().ends_with(b".remtori.com")
                    },
                ))
                .allow_methods([Method::GET]),
        )
        .layer(TraceLayer::new_for_http());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr).serve(app.into_make_service()).await.unwrap();
}

async fn handler(
    Extension(client): Extension<Client>,
    Query(params): Query<Params>,
    Path(path): Path<String>,
    req: Request<Body>,
) -> impl IntoResponse {
    let start = Instant::now();

    let resp = client
        .get(format!("https://cdn.remtori.com/{}", path))
        .send()
        .await
        .map_err(|err| {
            tracing::info!("Request error {err:#}");
            (StatusCode::NOT_FOUND, "Not Found")
        })?;

    if !resp.status().is_success() {
        tracing::info!("Request error: status code {}", resp.status());
        return Err((StatusCode::NOT_FOUND, "Not Found"));
    }

    let bytes = resp.bytes().await.map_err(|err| {
        tracing::error!("Request get bytes error {err:#}");

        (StatusCode::INTERNAL_SERVER_ERROR, "Decode response error")
    })?;

    tracing::info!(elapsed = start.elapsed().as_millis(), "Fetched image");
    let image = image::load_from_memory(&bytes[..]).map_err(|err| {
        tracing::error!("Decode image error {err:#}");
        (StatusCode::INTERNAL_SERVER_ERROR, "Decode image error")
    })?;

    tracing::info!(elapsed = start.elapsed().as_millis(), "Decoded image");
    let src_image = fir::Image::from_vec_u8(
        NonZeroU32::new(image.width()).unwrap(),
        NonZeroU32::new(image.height()).unwrap(),
        image.to_rgb8().into_raw(),
        fir::PixelType::U8x3,
    )
    .unwrap();

    let mut dst_image = fir::Image::new(
        NonZeroU32::new(image.width() / 4).unwrap(),
        NonZeroU32::new(image.height() / 4).unwrap(),
        src_image.pixel_type(),
    );

    let mut resizer = fir::Resizer::new(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
    if let Err(err) = resizer.resize(&src_image.view(), &mut dst_image.view_mut()) {
        tracing::error!({ uri = req.uri().path() }, "Resize image error {err:#}");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, "Resize image error"));
    }

    tracing::info!(elapsed = start.elapsed().as_millis(), "Resized image");
    let mut result_buf = Vec::new();
    JpegEncoder::new(&mut result_buf)
        .write_image(
            dst_image.buffer(),
            dst_image.width().get(),
            dst_image.height().get(),
            image::ColorType::Rgb8,
        )
        .unwrap();

    tracing::info!(elapsed = start.elapsed().as_millis(), "Encoded image");
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
}
