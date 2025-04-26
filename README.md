# Hotel Recommendation System

This project implements a memory-efficient hotel recommendation system using the HotelRec dataset, which contains approximately 50 million hotel reviews. The system combines different approaches to provide personalized hotel recommendations:
- Content-based filtering
- Collaborative filtering
- Hybrid approaches

## Key Features

- **Memory-Efficient Processing**: Handles 50GB+ dataset on machines with limited RAM
- **Chunked Data Loading**: Processes and trains on data chunks instead of loading entire dataset
- **Resumable Processing**: Supports interrupted processing with resume capability
- **Multi-Worker Support**: Efficient parallel data loading during training
- **Consistent Processing**: Maintains feature normalization and ID mappings across chunks

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

## System Requirements

Minimum requirements:
- 8GB RAM
- 100GB free disk space
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

Recommended:
- 16GB RAM
- 200GB free disk space
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM

## Data Preparation

Before training the models, prepare the dataset:

1. Download the HotelRec dataset and place `HotelRec.txt` in the `data/raw` directory.

2. Run the data preparation script:
   ```bash
   python src/utils/prepare_data.py --chunk-size 100000
   ```

   This will:
   - Process the raw data file in chunks
   - Create JSONL files in `data/processed`
   - Generate metadata for each chunk
   - Show processing statistics

   Options:
   - `--chunk-size`: Number of reviews per chunk (default: 100000)
   - `--force-clean`: Clean processed directory before starting
   - `--include-tfidf`: Include TF-IDF features
   - `--resume`: Resume from last processed chunk

3. The processed data will be available in `data/processed` directory.

### Random Sampling

For development or testing, create a smaller dataset sample:
```bash
python src/utils/prepare_random_sample.py --num-samples 2000000 --chunk-size 100000
```
This script randomly samples 2M reviews from HotelRec.txt, processes them similarly to prepare_data.py, and saves them in data/random_processed.

## Training Models

Train the recommendation models:

```bash
python src/utils/train_model.py --model-type content
python src/utils/train_model.py --model-type collaborative
python src/utils/train_model.py --model-type hybrid
```

Each model type uses memory-efficient training with:
- Chunked data loading
- Multi-worker processing
- Progress tracking
- Checkpointing
