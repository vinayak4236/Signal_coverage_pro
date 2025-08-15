import webbrowser
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import LinearColormap

from signal_model import (
    load_csv_dataset,
    generate_prediction_grid,
    SignalStrengthModel,
    build_signal_map,
)

DATA_PATH   = Path("d:/PROJECTS/SCMP/data/sample_measurements.csv")
OUTPUT_MAP  = Path("d:/PROJECTS/SCMP/signal_map.html")

COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
BINS   = [-120, -100, -90, -80, -70, -60, -40]

def make_colormap():
    return LinearColormap(colors=COLORS, vmin=BINS[0], vmax=BINS[-1],
                          caption="Signal strength (dBm)")

def get_browser_location() -> tuple[float, float]:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading, urllib.parse

    class LocHandler(BaseHTTPRequestHandler):
        lat = lon = None
        def do_GET(self):
            if self.path.startswith("/loc"):
                qs = urllib.parse.parse_qs(self.path.split("?", 1)[1])
                LocHandler.lat = float(qs["lat"][0])
                LocHandler.lon = float(qs["lon"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Location captured! You can close this tab.")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                html = """
                <!DOCTYPE html><html><head><meta charset="utf-8"/></head><body>
                <p>Allow location access when prompted…</p>
                <script>
                navigator.geolocation.getCurrentPosition(pos=>{
                    fetch(`/loc?lat=${pos.coords.latitude}&lon=${pos.coords.longitude}`);
                });
                </script></body></html>
                """
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

    with HTTPServer(("127.0.0.1", 0), LocHandler) as httpd:
        port = httpd.socket.getsockname()[1]
        url = f"http://localhost:{port}/"
        print("Opening browser to fetch accurate location …")
        webbrowser.open(url)
        httpd.serve_forever()
    return LocHandler.lat, LocHandler.lon


def main(model_type: str = "rf") -> None:
    df = load_csv_dataset(str(DATA_PATH))
    print(f"Loaded {len(df)} samples")
    model = SignalStrengthModel(model_type=model_type)
    result = model.fit(df)
    print(f"Trained {result.model_type}  R²={result.r2:.2f}  MAE={result.mae:.2f}")

    lat, lon = get_browser_location()
    print(f"Browser location: ({lat}, {lon})")

    grid = generate_prediction_grid(lat, lon, radius_m=2000, spacing_m=50)
    preds = model.predict(grid)
    grid["signal"] = preds

    # Detect column names
    lat_col = next(c for c in grid.columns if c.lower() in {"lat", "latitude"})
    lon_col = next(c for c in grid.columns if c.lower() in {"lon", "lng", "longitude"})
    sig_col = "signal"

    fmap = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")

    # Heat map
    heat_data = [[row[lat_col], row[lon_col], row[sig_col]] for _, row in grid.iterrows()]
    HeatMap(heat_data, min_opacity=0.4, radius=25, blur=15).add_to(fmap)

    # Colored circles
    cmap = make_colormap()
    for _, row in grid.iterrows():
        color = cmap(row[sig_col])
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
        ).add_to(fmap)

    # User marker & 2 km circle
    folium.Marker(
        location=[lat, lon],
        popup="You are here",
        icon=folium.Icon(color="black", icon="user", prefix="fa"),
    ).add_to(fmap)
    folium.Circle(
        location=[lat, lon],
        radius=2000,
        color="#0078ff",
        weight=2,
        fill=False,
    ).add_to(fmap)

    cmap.add_to(fmap)
    folium.LayerControl().add_to(fmap)

    fmap.save(str(OUTPUT_MAP))
    webbrowser.open(str(OUTPUT_MAP))


if __name__ == "__main__":
    main("rf")