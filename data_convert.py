import pyreadstat

# Step 2: Define file paths
input_file = "CGSS2021.sav"  # Replace with your actual path
output_file = "CGSS2021_converted.csv"

# Step 3: Read the SPSS .sav file
df, meta = pyreadstat.read_sav(input_file)

# Step 4: Export to CSV
df.to_csv(output_file, index=False)

print(f"File successfully converted and saved as: {output_file}")