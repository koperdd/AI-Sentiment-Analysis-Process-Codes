import pandas as pd
import zstandard as zstd
import json
import os
from tqdm import tqdm

# File paths
input_file = 'E:/submissions/RS_2024-12.zst'
output_dir = 'E:/Reddit_2024'  # Desired output directory

# Extract date from the input file name
input_filename = os.path.basename(input_file)
date_from_file = input_filename.split('_')[1].split('.')[0]  # Extract "2024-12"

# List of subreddits to filter
target_subreddits = {"Singapore"}  # Subreddit to filter

# Generate output file name based on the subreddit and extracted date
subreddit_name = list(target_subreddits)[0]  # Get the name of the first subreddit in the set
output_file = os.path.join(output_dir, f"{subreddit_name}submissions_{date_from_file}.json")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define chunk size in bytes
chunk_size = 10 * 1024 * 1024  # 10 MB per chunk

# Estimate total lines by sampling the beginning of the file
print("Estimating total lines...")
total_size = os.path.getsize(input_file)
sample_size = 1024 * 1024  # 1 MB sample

with open(input_file, 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed_file) as reader:
        sample_data = reader.read(sample_size)
        sample_lines = sample_data.split(b'\n')
        avg_line_size = sum(len(line) for line in sample_lines if line) / len(sample_lines)
        estimated_total_lines = int(total_size / avg_line_size)
        print(f"Estimated total lines: {estimated_total_lines}")

print("Starting the decompression and filtering...")

# Open the .zst file and decompress in streaming mode
with open(input_file, 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed_file) as reader:
        data = []  # Temporary list to hold JSON objects
        buffer = b""  # Buffer to accumulate decompressed data
        total_lines = 0  # Counter to track the total lines processed
        filtered_lines = 0  # Counter for lines matching the target subreddits

        # Open the output file to write JSON lines
        with open(output_file, 'w', encoding='utf-8') as outfile:
            chunk_count = 1  # Track the number of chunks processed

            # Initialize tqdm progress bar
            with tqdm(total=estimated_total_lines, desc="Processing lines", unit="lines") as pbar:
                while True:
                    # Read a chunk of data
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        pbar.set_description("Processing complete!")
                        break  # End of file

                    # Decode chunk and add to buffer
                    buffer += chunk

                    # Split buffer into lines based on newline character
                    lines = buffer.split(b'\n')

                    # Process all lines except the last one (which may be incomplete)
                    for line in lines[:-1]:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            total_lines += 1

                            # Filter based on the subreddit field only
                            if json_line.get('subreddit') in target_subreddits:
                                data.append(json_line)
                                filtered_lines += 1

                            # Update the progress bar
                            pbar.update(1)

                            # Write in chunks of JSON lines to save memory
                            if len(data) >= 10000:
                                pbar.set_description(f"Writing {len(data)} records to JSON file...")
                                df = pd.DataFrame(data)
                                df.to_json(outfile, orient='records', lines=True)
                                outfile.write('\n')
                                data = []  # Clear data for the next batch
                        except json.JSONDecodeError:
                            # Skip malformed lines (quietly)
                            continue

                    # Keep the last (possibly incomplete) line in buffer for next chunk
                    buffer = lines[-1]
                    chunk_count += 1

                # Write any remaining data
                if data:
                    pbar.set_description(f"Writing remaining {len(data)} records to JSON file...")
                    df = pd.DataFrame(data)
                    df.to_json(outfile, orient='records', lines=True)
                    outfile.write('\n')

print(f"Decompression and filtering complete.")
print(f"Total lines processed: {total_lines}")
print(f"Total lines matching subreddits: {filtered_lines}")
print(f"Filtered data has been saved to {output_file}")
