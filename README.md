# Paryavaran — Urban Greenspace Suitability Dashboard (Delhi, India)

## Inspiration
Living in cities, we appreciate the importance of green space in our areas, especially as rapid urbanization continues. However, since neither of us are city planners or subject matter experts, we thought it would be most valuable to develop a flexible tool. That way a city planner can integrate other relevant datasets and keep it updated with recent readings to dynamically plan green infrastructure where it's needed most.

## Overview
"Today, 56% of the world's population - 4.4 billion inhabitants - live in cities. This trend is expected to continue. By 2050, with the urban population more than doubling its current size, nearly 7 of 10 people in the world will live in cities." - The World Bank

Green spaces improve both the environmental and social conditions of cities. Air quality, population satisfaction, urban temperatures, biodiversity, flood risk reduction, noise abatement - these are among the major benefits of greenspaces in an urban area. We decided to target **Delhi (NCT)**, one of the most densely populated and polluted metropolitan areas in the world, to see how open data can inform the locations for green spaces that would maximize their positive effects on these urban conditions.

We extracted raw data from multiple open sources—ranging from WorldPop demographic estimates to NASA MODIS satellite data and Central Pollution Control Board (CPCB) monitoring stations. We processed them by upsampling to a high enough resolution (250m) that makes the dashboard highly usable by city planners. The resolution itself is a parameter, with distance-weighted k-nearest neighbours (KNN) and vectorized spatial joins enabling high-performance processing across the city grid.

The final dashboard aggregates the underlying data into a **Priority Score** that summarizes where potential greenspaces would most benefit the urban conditions of Delhi. 

The calculation of the score considers multiple weighted components:
- **Population Need (0.25):** Based on WorldPop India 2024 raster aggregations.
- **Greenspace Deficit (0.20):** Based on existing parks from OpenStreetMap (OSM) and ESA WorldCover.
- **Heat Stress (0.20):** Derived from MODIS Summer Mean Land Surface Temperature.
- **Air Pollution (0.15):** Interpolated from real-time CPCB ground stations (PM2.5, PM10, NO2).
- **Accessibility Deficit (0.10):** Distance to nearest roads.
- **Building Density (0.10):** Google Open Buildings V3 footprint fractions.

A cornerstone of our project was using optimized K-Nearest Neighbour models and Vector Intersections (via Geopandas) to project all datasets onto a single 2D grid. We handled spherical distances using the Haversine formula facilitated by SKLearn, mapping complex geographical data perfectly to the coordinates of Delhi.

Using OpenStreetMap features, we were able to filter land feasibility—ensuring we do not recommend building a park on top of a hospital or an airport, or inside an existing water body.

## Dashboard Screenshots

### 🏆 Priority Score Map (Delhi)
![Priority Score Map](Dashboard_Images/Priority_Score.png)

### 📊 Air Quality Score
![Air Quality Map](Dashboard_Images/Air_Quality_Score.png)

### 📊 Population Density
![Population Density](Dashboard_Images/Population_Density.png)

### 📊 PM2.5 Concentration
![PM2.5 Map](Dashboard_Images/PM25.png)

### 📊 NO2 Concentration
![NO2 Map](Dashboard_Images/NO2.png)


## UN Sustainable Development Goals
We meet the following UN Sustainable Development Goals:
- **Good health and Well-Being:** Green spaces protect the local populace from toxic gases and high urban temperatures but also provide a mental benefit by being a place of community.
- **Reduced inequalities:** Considering the wide-ranging benefits of green spaces to people's health, our dashboard helps solve 'greenspace inequality'.
- **Sustainable Cities and Communities:** Green spaces help cultivate communities in cities by providing a space for activities.
- **Climate Action:** Green spaces contribute to the absorption of greenhouse gases and reduced surface temperatures (Urban Heat Island effect).
- **Life on Land:** Green spaces in urban metropolises protect biodiversity by allowing space for ecosystems to thrive.
- **Partnerships for the Goals:** Given the wide-ranging data employed (ESA, NASA, CPCB, Google), maintaining this requires a coalition of parties.

## Future Work
Besides obviously expanding the datasets and getting true domain experts to refine the greenspace calculation, we see an expansion of scope of this project to envelope other major cities of the world, like Mumbai or Bangalore. Once this is achieved, further aggregated analysis can be provided on the dashboard such as relative metrics to compare cities.

Our resolution of 250m is a parameter. The vectorized processing pipeline facilitates any resolution, however, ideally, one would prefer to add higher resolution data to begin with so that predictions do not deviate from reality.

We have demonstrated our dashboard via Plotly Dash, which suited our needs for a Minimum Viable Product, but given more time, we would implement our dashboard in a more scalable cloud infrastructure environment to handle global datasets without bottlenecking.

## How we built it
We started off by scanning through the datasets suitable for India and Delhi specifically. We narrowed it down to CPCB Air Quality data, ESA WorldCover, MODIS Heat data, Google Open Buildings, and WorldPop. 

We used Python libraries (`geopandas`, `scipy`, `sklearn`, `rasterio`) for EDA, data preprocessing, visualization, and model building. We built optimized Python scripts that automate the generation of a 250m bounding grid over Delhi, execute vectorized spatial joins against thousands of OpenStreetMap polygons, and run KNN interpolation for pollution nodes.

To visualize the data on our Plotly Dash dashboard, we load the final aggregated CSV that serves as the single source of truth for the city grid.

## Technologies
- Python (Jupyter, Pandas, Geopandas, Scipy, Sklearn, Rasterio, Osmnx)
- Plotly Dash for Frontend Visualization
- OpenStreetMap / ESA WorldCover / NASA MODIS / CPCB
- GitHub for Version Control

## Accomplishments that we're proud of
The main accomplishments were around data preprocessing and optimization. We heavily refactored the original algorithms to use vectorized `gpd.sjoin` and advanced indexing. We really had to dig into the details of coordinate reference systems (CRS) to perfectly project satellite raster data and vector polygons onto a unified Haversine grid.

There was also a lot of effort that went into optimizing the code so we could process at higher resolutions significantly faster—what used to take $O(N^2)$ time looping over map features was reduced to seconds.

Finally, we're proud to deliver a fully open-source, open-data solution tailored for Delhi that can be used flexibly to influence real-world urban planning decision making.

## How to setup and run
To setup the fundamental CSVs, create a `Data/` directory and place the downloaded rasters (`worldpop_india_2024_100m.tif`, `worldcover_delhi.tif`, `modis_lst_summer_mean.tif`) and vectors (`open_buildings_delhi.gpkg`, `cpcb_stations.csv`) into it. OpenStreetMap data is fetched dynamically via Osmnx.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Pipeline:**
   ```bash
   # Step 1 – build grid and feature CSV
   python Scripts/build_land_type_csv.py
   
   # Step 2 & 3 – generate penultimate and final dataframes
   jupyter nbconvert --to notebook --execute Notebooks/create_penultimate_df.ipynb
   jupyter nbconvert --to notebook --execute Notebooks/create_final_df.ipynb
   ```
3. **Launch dashboard:**
   ```bash
   cd Plotly_Dash_App
   python app.py
   ```
The dashboard runs at `http://127.0.0.1:8050`. To edit the green space score function, enter the `helpers.py` script where the core logic resides.

## Built With
- dash
- geopandas
- pandas
- plotly
- python
- scikit-learn
- jupyter
- rasterio
- osmnx
