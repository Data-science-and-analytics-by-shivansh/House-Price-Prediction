import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
import folium
from shapely.geometry import Point

def add_geospatial_features(df):
    # Simulate lat/long if not present (in real: use API or dataset with coords)
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        df['Latitude'] = np.random.uniform(42.0, 43.0, len(df))  # Ames, IA approx
        df['Longitude'] = np.random.uniform(-93.7, -93.5, len(df))
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Clustering locations (e.g., 5 clusters for neighborhoods)
    coords = df[['Latitude', 'Longitude']]
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['LocationCluster'] = kmeans.fit_predict(coords)
    
    # Proximity features (e.g., distance to city center)
    city_center = Point(-93.6198, 42.0308)  # Ames center
    gdf['DistToCenter'] = gdf.geometry.distance(city_center)
    
    # Add more: Proximity to parks, schools (in real: use external data)
    
    return df

def visualize_map(df):
    m = folium.Map(location=[42.0308, -93.6198], zoom_start=12)
    for idx, row in df.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=f"Price: ${row['SalePrice']}").add_to(m)
    m.save('geospatial_map.html')

# Example
if __name__ == "__main__":
    df = pd.read_csv('../data/engineered_train.csv')
    geo_df = add_geospatial_features(df)
    visualize_map(geo_df)
