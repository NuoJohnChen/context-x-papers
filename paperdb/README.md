Build ArXiv Datasets and Get Embeddings.

```
conda activate arxivdata
requirements: kagglehub

# download metadata
#https://www.kaggle.com/datasets/Cornell-University/arxiv 
/shared/hdd/nuochen/arxivdata/kaggle/arxiv-metadata-oai-snapshot.json


# fields like
2387930:{"id":"2508.04575",
"submitter":"Nuo Chen",
"authors":"Nuo Chen, Yicheng Tong, Jiaying Wu",
"title":"Beyond Brainstorming",
"comments":"Preprint",
"journal-ref":null,"doi":null,"report-no":null,
"categories":"cs.CL cs.AI cs.CY",
"license":"http://creativecommons.org/licenses/by/4.0/",
"abstract":"While AI a",
"versions":
[{"version":"v1","created":"Wed, 06 Aug 2025 15:59:18 GMT"}],
"update_date":"2025-08-07",
"authors_parsed":[["Chen","Nuo",""],["Tong","Yicheng",""],["Wu","Jiaying",""]}

# 2. Update and fetch all PDFs from Jan-Aug 2025. parse2pdf (incrementally add to /shared/hdd/arxiv_pdfs)

python /shared/hdd/nuochen/arxivdata/download_arxiv_pdfs.py
Only for 2025: /shared/hdd/nuochen/arxivdata/download_and_extract_arxiv.py
#python3 download_arxiv_pdfs.py --meta-file /path/meta.jsonl --output-dir /path/pdfs --limit 1000 --sleep 0.2 --retries 3 --progress

# (2*)
If you need to continue, because direct pdflink download may result in some corrupted PDFs (this step is not complete), use arxiv api for those:
# Step 1: Only scan and save the list of corrupted IDs (do not download)
python3 redownload_corrupted_25_pdfs.py --dry-run --progress
# Step 2: Redownload the corrupted PDF files
python3 redownload_corrupted_25_pdfs.py --progress
# If already scanned, you can directly load the ID list to redownload
python3 redownload_corrupted_25_pdfs.py --load-ids --progress
/shared/hdd/nuochen/arxivdata/reconvert_corrupted_mmds.py

# (2^ Full download)
(For full download use:
#1. Download PDF data
/shared/hdd/nuochen/arxivdata/download_and_extract_arxiv_alltime.py
)
/shared/hdd/arxiv_pdfs

# The above process can be fixed to run directly:
python /shared/hdd/nuochen/arxivdata/redownload_corrupted_pdfs_byarxiv.py --year 2x --progress


#3. pdf to mmd
# https://github.com/facebookresearch/nougat
# pymupdf
#python /shared/hdd/nuochen/arxivdata/extract_first_page_mmd.py
python3 batch_convert_25_pdfs.py --workers 64 --progress
/shared/hdd/arxiv_mmds

# Use GPT to traverse the first page and add "affiliation", "department" (if any), "countries" (on a100)

## Two ways to get affiliation: one from semantic scholar by directly getting the organization's papers, the other by extracting from PDF

# 4. json2jsonl update fields
python /shared/hdd/nuochen/arxivdata/scripts/build_addaffiliation_from_kaggle_25.py

# Add to jsonl file: /shared/hdd/nuochen/arxivdata/index/specter2_merged/meta.enriched.clean.final_inline.fast.enhanced.deduplicated_by_first_author.jsonl
# Incremental file: /shared/hdd/nuochen/arxivdata/index/specter2_merged/addaffiliation.jsonl
{
  "id": "Arizona_State_University_0",
  "university": "Arizona_State_University",
  "title": "Mergers, Radio Jets, and Quenching Star Formation in Massive Galaxies: Quantifying Their Synchronized Cosmic Evolution and Assessing the Energetics",
  "abstract": "The existence of a population of massive quiescent galaxies with little to no star formation poses a challenge to our understanding of galaxy evolution. The physical process that quenched the star formation in these galaxies is debated, but the most popular possibility is that feedback from supermassive black holes lifts or heats the gas that would otherwise be used to form stars. In this paper, we evaluate this idea in two ways. First, we compare the cumulative growth in the cosmic inventory of the total stellar mass in quiescent galaxies to the corresponding growth in the amount of kinetic energy carried by radio jets. We find that these two inventories are remarkably well-synchronized, with about 50% of the total amounts being created in the epoch from z ≈ 1 to 2. We also show that these agree extremely well with the corresponding growth in the cumulative number of major mergers that result in massive (>1011 M ʘ) galaxies. We therefore argue that major mergers trigger the radio jets and also transform the galaxies from disks to spheroids. Second, we evaluate the total amount of kinetic energy delivered by jets and compare it to the baryonic binding energy of the galaxies. We find the jet kinetic energy is more than sufficient to quench star formation, and the quenching process should be more effective in more massive galaxies. We show that these results are quantitatively consistent with recent measurements of the Sunyaev–Zel'dovich effect seen in massive galaxies at z ≈ 1.",
  "text": "Mergers, Radio Jets, and Quenching Star Formation in Massive Galaxies: Quantifying Their Synchronized Cosmic Evolution and Assessing the Energetics[SEP]The existence of a population of massive quiescent galaxies with little to no star formation poses a challenge to our understanding of galaxy evolution...",
  "authors": ["Timothy Heckman", "Namrata Roy", "Philip Best", "Rohit Kondapally"],
  "categories": ["astro-ph.GA"],
  "year": 2024,
  "arxiv_id": "2410.09157",
  "paperId": "1e7de1cfbc4d38dc9410d252ec88c327bf6e8f26",
  "authorId": "2264685984",
  "hIndex": 5,
  "first_author": "Timothy Heckman"
}
python /shared/hdd/nuochen/arxivdata/scripts/fill_affiliations_from_mmd_gpt.py --progress



Final data:
{
  "id": "",
  "university": "",
  "title": <original title>,
  "abstract": <original abstract>,
  "text": "<title>[SEP]<abstract>",
  "authors": ["First Last", ...],  // parsed from original authors (comma-separated string)
  "categories": ["cs.CL", "cs.AI", ...], // split by whitespace
  "year": <int>,             // derive from update_date or versions[v1].created
  "arxiv_id": <original id>,
  "paperId": "",
  "authorId": "",
  "hIndex": "",
  "first_author": "First Last",   // first name in authors list
  "published_date": "YYYY-MM-DD", // from update_date (preferred) or v1 created date
  "affiliation": "",
  "department": "",
  "affiliation_country": ""
}




# Merge data and remove duplicates
!!!! Please make sure to add the last several fields to /shared/hdd/nuochen/arxivdata/index/specter2_merged/meta.enriched.clean.final_inline.fast.enhanced.deduplicated_by_first_author.jsonl first !!!!



# 5. json2npy word embedding
#npy
4. Generate SPECTER2 word embedding vectors
Script: scripts/embed_specter2.py
Input: arxiv_merged.jsonl 
Process: Use SPECTER2 model to generate 768-dimensional word embeddings (can use GPU)
Output:
arxiv_merged.pt (PyTorch format)
arxiv_merged.npy (NumPy format)
arxiv_merged.jsonl (metadata)
arxiv_merged.json (processed abstracts)

# 6. json2parquet JSONL to Parquet format
Script: tools/jsonl_to_parquet.py
Input: arxiv_merged.jsonl 
Process: Extract specified fields and convert to Parquet format
Output: arxiv_merged.parquet

# Migrate to web
#/home/nuochen/workshop_reco/index/specter2_merged/embedding_atlas_export_20250829_101535/data/dataset_with_published_with_published_date_update_with_published_date_update.parquet",
#/home/nuochen/workshop_reco/outputs/minilm_filtered/embeddings.npy

# Replace "versions":
[{"version":"v1","created":"Wed, 06 Aug 2025 15:59:18 GMT"}] with published_date


# Sequentially add published_date, update parquet




```
