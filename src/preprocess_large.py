import library.preprocessor as preprocessor

src_language = "eng"
tgt_language = "fra"

train_file = f"{src_language}_{tgt_language}_large_train.csv"
val_file = f"{src_language}_{tgt_language}_large_val.csv"
test_file = f"{src_language}_{tgt_language}_large_test.csv"
src_max_len = 55
tgt_max_len = 69

print("Loading data...")
train_dataloader = preprocessor.DataLoader(f"../data/{train_file}")
src_train_texts, tgt_traintexts = train_dataloader.load(header=1)
val_dataloader = preprocessor.DataLoader(f"../data/{val_file}")
src_val_texts, tgt_val_texts = val_dataloader.load(header=1)
test_dataloader = preprocessor.DataLoader(f"../data/{test_file}")
src_test_texts, tgt_test_texts = test_dataloader.load(header=1)
print("Done loading data. Number of sentences in training set:", len(src_train_texts))
print("Number of sentences in validation set:", len(src_val_texts))
print("Number of sentences in test set:", len(src_test_texts), "\n\n")
print("First 10 training source language sentences:", src_train_texts[:10], "\n\n")
print("First 10 training target language sentences:", tgt_traintexts[:10], "\n\n")
print("First 10 validation source language sentences:", src_val_texts[:10], "\n\n")
print("First 10 validation target language sentences:", tgt_val_texts[:10], "\n\n")
print("First 10 test source language sentences:", src_test_texts[:10], "\n\n")
print("First 10 test target language sentences:", tgt_test_texts[:10], "\n\n")


print("Standardizing texts...")
standardizer = preprocessor.Standardizer()
standadized_src_train_texts = standardizer.standardize(src_train_texts)
standadized_tgt_train_texts = standardizer.standardize(tgt_traintexts)
standadized_src_val_texts = standardizer.standardize(src_val_texts)
standadized_tgt_val_texts = standardizer.standardize(tgt_val_texts)
standadized_src_test_texts = standardizer.standardize(src_test_texts)
standadized_tgt_test_texts = standardizer.standardize(tgt_test_texts)
print("Done standardizing texts.")
print("First 10 standardized training source language sentences:", standadized_src_train_texts[:10], "\n\n")
print("First 10 standardized training target language sentences:", standadized_tgt_train_texts[:10], "\n\n")
print("First 10 standardized validation source language sentences:", standadized_src_val_texts[:10], "\n\n")
print("First 10 standardized validation target language sentences:", standadized_tgt_val_texts[:10], "\n\n")
print("First 10 standardized test source language sentences:", standadized_src_test_texts[:10], "\n\n")
print("First 10 standardized test target language sentences:", standadized_tgt_test_texts[:10], "\n\n")


print("Tokenizing texts...")
tokenizer = preprocessor.Tokenizer()
src_train_tokens = tokenizer.word_tokenize(standadized_src_train_texts, src_max_len)
tgt_train_tokens = tokenizer.word_tokenize(standadized_tgt_train_texts, tgt_max_len)
src_val_tokens = tokenizer.word_tokenize(standadized_src_val_texts, src_max_len)
tgt_val_tokens = tokenizer.word_tokenize(standadized_tgt_val_texts, tgt_max_len)
src_test_tokens = tokenizer.word_tokenize(standadized_src_test_texts, src_max_len)
tgt_test_tokens = tokenizer.word_tokenize(standadized_tgt_test_texts, tgt_max_len)
print("Done tokenizing texts.")
print("Number of training source language sentences:", len(src_train_tokens))
print("Number of training target language sentences:", len(tgt_train_tokens))
print("First 10 training source language tokenized sentences:", src_train_tokens[:10], "\n\n")
print("First 10 training target language tokenized sentences:", tgt_train_tokens[:10], "\n\n")
print("Number of validation source language sentences:", len(src_val_tokens))
print("Number of validation target language sentences:", len(tgt_val_tokens))
print("First 10 validation source language tokenized sentences:", src_val_tokens[:10], "\n\n")
print("First 10 validation target language tokenized sentences:", tgt_val_tokens[:10], "\n\n")
print("Number of test source language sentences:", len(src_test_tokens))
print("Number of test target language sentences:", len(tgt_test_tokens))
print("First 10 test source language tokenized sentences:", src_test_tokens[:10], "\n\n")
print("First 10 test target language tokenized sentences:", tgt_test_tokens[:10], "\n\n")


print("Indexing source language texts...")
src_indexer = preprocessor.Indexer()
src_indexer.build_vocab(src_train_tokens)
print("Done indexing source language texts. ")
print("Vocabulary size:", src_indexer.vocab_size)
print("First 10 words in source language vocabulary:", list(src_indexer.word2idx.keys())[:10], "\n\n")

print("Indexing target language texts...")
tgt_indexer = preprocessor.Indexer()
tgt_indexer.build_vocab(tgt_train_tokens)
print("Done indexing target language texts. ")
print("Vocabulary size:", tgt_indexer.vocab_size)
print("First 10 words in target language vocabulary:", list(tgt_indexer.word2idx.keys())[:10], "\n\n")
print("Saving vocabulary...")
src_indexer.save_vocab(f"../data/{src_language}_vocab_large.pkl")
tgt_indexer.save_vocab(f"../data/{tgt_language}_vocab_large.pkl")
print("Done saving vocabulary to ../data/{src_language}_vocab.pkl and ../data/{tgt_language}_vocab.pkl.\n\n")


print("Converting source language tokens to indices...")
src_train_indices = src_indexer.text_to_indices(src_train_tokens)
src_val_indices = src_indexer.text_to_indices(src_val_tokens)
src_test_indices = src_indexer.text_to_indices(src_test_tokens)
print("Done converting source language tokens to indices. First 10 sentences:", src_train_indices[:10], "\n\n")
print("Converting target language tokens to indices...")
tgt_train_indices = tgt_indexer.text_to_indices(tgt_train_tokens, prepend_sos=True)
tgt_val_indices = tgt_indexer.text_to_indices(tgt_val_tokens, prepend_sos=True)
tgt_test_indices = tgt_indexer.text_to_indices(tgt_test_tokens, prepend_sos=True)
print("Done converting target language tokens to indices. First 10 sentences:", tgt_train_indices[:10], "\n\n")


splitter = preprocessor.Splitter()

print("Saving training data...")
splitter.save_indices(src_train_indices, f"../data/{src_language}_train_large.pt")
splitter.save_indices(tgt_train_indices, f"../data/{tgt_language}_train_large.pt")
print(f"Done saving training data to ../data/{src_language}_train_large.pt and ../data/{tgt_language}_train_large.pt.\n\n")

print("Saving validation data...")
splitter.save_indices(src_val_indices, f"../data/{src_language}_val_large.pt")
splitter.save_indices(tgt_val_indices, f"../data/{tgt_language}_val_large.pt")
print(f"Done saving validation data to ../data/{src_language}_val_large.pt and ../data/{tgt_language}_val_large.pt.\n\n")

print("Saving test data...")
splitter.save_indices(src_test_indices, f"../data/{src_language}_test_large.pt")
splitter.save_indices(tgt_test_indices, f"../data/{tgt_language}_test_large.pt")
print(f"Done saving test data to ../data/{src_language}_test_large.pt and ../data/{tgt_language}_test_large.pt.\n\n")