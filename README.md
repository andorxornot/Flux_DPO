# Flux DPO

This project implements Direct Preference Optimization (DPO) for FLUX. DPO is a technique for aligning language models with human preferences without using reward models.

## Script Execution Sequence

1. `generate_images.py` - Generates initial images for the dataset
2. `process_images.py` - Processes and prepares the generated images
3. `create_dpo_dataset.py` - Creates the dataset for DPO training
