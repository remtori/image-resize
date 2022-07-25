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
) -> impl IntoResponse {
    let start = Instant::now();

    let resp = client
        .get(format!("https://cdn.remtori.com/{}", path))
        .send()
        .await
        .map_err(|err| {
            tracing::info!(path, "Request error {err:#}");
            (StatusCode::NOT_FOUND, "Not Found")
        })?;

    if !resp.status().is_success() {
        tracing::info!(path, "Request error: status code {}", resp.status());
        return Err((StatusCode::NOT_FOUND, "Not Found"));
    }

    let bytes = resp.bytes().await.map_err(|err| {
        tracing::error!(path, "Request get bytes error {err:#}");
        (StatusCode::INTERNAL_SERVER_ERROR, "Decode response error")
    })?;

    let time_fetch = start.elapsed();
    let start = Instant::now();
    let image = image::load_from_memory(&bytes[..]).map_err(|err| {
        tracing::error!(path, "Decode image error {err:#}");
        (StatusCode::INTERNAL_SERVER_ERROR, "Decode image error")
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
        if let (Some(width), Some(height)) = (params.width, params.height) {
            (width.get(), height.get())
        } else if let Some(width) = params.width {
            let ratio = width.get() as f32 / src_image.width().get() as f32;
            (width.get(), (src_image.height().get() as f32 * ratio) as u32)
        } else if let Some(height) = params.height {
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

    let mut resizer = fir::Resizer::new(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
    if let Err(err) = resizer.resize(&src_image.view(), &mut dst_image.view_mut()) {
        tracing::error!(path, "Resize image error {err:#}");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, "Resize image error"));
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
}
