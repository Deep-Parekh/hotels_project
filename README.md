# Hotel Recommendation System

This project implements a hotel recommendation system using the HotelRec dataset, which contains approximately 50 million hotel reviews. The system combines different approaches to provide personalized hotel recommendations:
- Content-based filtering
- Collaborative filtering
- Hybrid approaches

## Dataset

The project uses the HotelRec dataset, which includes:
- ~50 million hotel reviews
- Multiple features per review:
  - Overall ratings (1-5 scale)
  - Sub-ratings (sleep quality, value, rooms, service, cleanliness, location)
  - Text reviews with titles
  - User and hotel metadata
  - Temporal information

Dataset source: [HotelRec GitHub Repository](https://github.com/Diego999/HotelRec)

## Data Preparation

Before training the models, you need to prepare the dataset:

1. Download the HotelRec dataset and place `HotelRec.txt` in the `data/raw` directory.

2. Run the data preparation script:
   ```bash
   python src/utils/prepare_data.py
   ```

   This will:
   - Process the raw data file
   - Create chunked parquet files in `data/processed`
   - Show processing statistics

3. The processed data will be available in `data/processed` directory.

