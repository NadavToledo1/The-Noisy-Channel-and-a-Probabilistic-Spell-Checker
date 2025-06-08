import collections
import math
import random
import re
import string
import time
import nltk


# --- Import necessary components from other files ---
try:
    # Assuming Spell_Checker class is in ex1.py
    from ex1 import Spell_Checker
    # Assuming error_tables dict is in spelling_confusion_matrices.py
    from spelling_confusion_matrices import error_tables as raw_error_tables
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Make sure ex1.py and spelling_confusion_matrices.py exist and contain the required classes/variables.")
    exit()

# --- NLTK Download Check ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NOTE: NLTK 'punkt' data not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' downloaded.")


# --- Normalization Function ---
def normalize_text(text):
    """Returns a normalized version of the specified string."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Error Generation Utility Functions ---


def _apply_edit(word, position, edit_type, details):
    """Applies a single specified edit to a word."""
    if not word and edit_type != 'insertion': return "" # Cannot edit empty word except insertion
    if edit_type == 'substitution':
        if position < len(word): return word[:position] + details[1] + word[position+1:]
    elif edit_type == 'deletion':
        if position < len(word): return word[:position] + word[position+1:]
    elif edit_type == 'insertion':
        return word[:position] + details[1] + word[position:]
    elif edit_type == 'transposition':
        if position + 1 < len(word): return word[:position] + word[position+1] + word[position] + word[position+2:]
    return word


def _choose_specific_edit(word, position, edit_type, error_tables):
    """Chooses a specific edit based on context and error_tables counts."""
    relevant_table = error_tables.get(edit_type, {})
    if not relevant_table: return None
    options, weights = [], []
    if edit_type == 'substitution':
        if position < len(word):
            intended_char = word[position]
            for key, count in relevant_table.items():
                if isinstance(key, str) and len(key) == 2 and key[0] == intended_char:
                    options.append((intended_char, key[1])); weights.append(count)
    elif edit_type == 'deletion':
        if position < len(word):
            char_deleted = word[position]; char_before = word[position-1] if position > 0 else '#'
            key_str = f"{char_before}{char_deleted}"
            count = relevant_table.get(key_str, 0)
            if count > 0: options.append((char_before, char_deleted)); weights.append(count)
    elif edit_type == 'insertion':
         char_before = word[position-1] if position > 0 else '#'
         for key, count in relevant_table.items():
             if isinstance(key, str) and len(key) == 2 and key[0] == char_before:
                  options.append((char_before, key[1])); weights.append(count)
    elif edit_type == 'transposition':
         if position + 1 < len(word):
              char1, char2 = word[position], word[position+1]
              key_str = f"{char1}{char2}"
              count = relevant_table.get(key_str, 0)
              if count > 0: options.append((char1, char2)); weights.append(count)
    if not options or not weights or sum(weights) <= 0: return None
    return random.choices(options, weights=weights, k=1)[0]


def _choose_edit_type(error_tables):
    """Chooses an edit type based on total counts for each type."""
    total_counts = {k: sum(error_tables.get(k, {}).values()) for k in ['deletion', 'insertion', 'substitution', 'transposition']}
    types, counts = list(total_counts.keys()), list(total_counts.values())
    if not types or not counts or sum(counts) <= 0: return None
    return random.choices(types, weights=counts, k=1)[0]


def _apply_probabilistic_edits(word, num_edits, error_tables):
    """Applies 1 or 2 probabilistic edits to a word."""
    current_word = word
    edits_applied = 0
    edit_type = '' # Define edit_type before the loop
    while edits_applied < num_edits:
        allow_insertion = (not current_word and edits_applied == 0)
        if not current_word and not allow_insertion: break

        edit_type = _choose_edit_type(error_tables)
        if not edit_type: break

        max_pos = len(current_word) if edit_type != 'insertion' else len(current_word) + 1
        if max_pos == 0 and edit_type != 'insertion': break
        position = random.randrange(max_pos)

        edit_details = _choose_specific_edit(current_word, position, edit_type, error_tables)
        if not edit_details: break

        new_word = _apply_edit(current_word, position, edit_type, edit_details)
        if new_word != current_word:
            current_word = new_word
            edits_applied += 1
        else: break
    return current_word


def generate_sentence_with_error(sentence, error_tables, max_edits=2, prob_dist2=0.1):
    """Generates a sentence pair: (original, with_one_realistic_error)."""
    original_sentence = sentence
    original_tokens = original_sentence.split()
    if not original_tokens: return (original_sentence, original_sentence)
    max_attempts = 5
    suitable_word_found = False
    word_index, word_to_corrupt = -1, ""
    for _ in range(max_attempts):
        idx = random.randrange(len(original_tokens))
        word = original_tokens[idx]
        if word and re.match(r'.*[a-z]', word):
             word_index, word_to_corrupt = idx, word
             suitable_word_found = True
             break
    if not suitable_word_found: return (original_sentence, original_sentence)
    num_edits = 1
    if max_edits == 2 and random.random() < prob_dist2: num_edits = 2
    final_corrupted_word = _apply_probabilistic_edits(word_to_corrupt, num_edits, error_tables)
    erroneous_tokens = list(original_tokens)
    erroneous_tokens[word_index] = final_corrupted_word
    erroneous_sentence = " ".join(erroneous_tokens)
    return (original_sentence, erroneous_sentence) if erroneous_sentence != original_sentence else (original_sentence, original_sentence)


# # --- Main Evaluation Block ---
# if __name__ == "__main__":
#
#     # +++ Set Random Seed for Reproducibility +++
#     SEED_VALUE = 15 # You can choose any integer
#     random.seed(SEED_VALUE)
#     # Note: NLTK's sentence tokenizer might have its own randomness if models vary,
#     # but the core Python random operations will now be fixed.
#     # ++++++++++++++++++++++++++++++++++++++++++++
#
#     print("Starting Spell Checker Evaluation...")
#     print(f"(Using random seed: {SEED_VALUE})")
#
#     # --- Configuration ---
#     corpus_file_path = 'big.txt'
#     lm_ngram_n = 3
#     evaluation_set_size = 100 # Keep small for faster debugging runs
#     error_gen_prob_dist2 = 0.25
#     # Let's try a slightly lower alpha as discussed
#     spell_check_alpha = 0.7 # Testing 0.90
#
#     # --- Load Error Tables (Imported) ---
#     try:
#         # Ensure raw_error_tables is accessible
#         if not isinstance(raw_error_tables, dict) or not raw_error_tables:
#              print("Error: raw_error_tables is empty or not a dictionary.")
#              exit()
#         print(f"Loaded raw error tables (Number of types: {len(raw_error_tables)})")
#     except NameError:
#         print("Error: raw_error_tables not found. Make sure it's imported from error_tables.py")
#         exit()
#
#     # --- Load Corpus Data ---
#     try:
#         with open(corpus_file_path, 'r', encoding='utf-8') as f: corpus_text = f.read()
#         print(f"Loaded corpus from {corpus_file_path} ({len(corpus_text)} chars)")
#     except FileNotFoundError: print(f"Error: Corpus file not found at {corpus_file_path}"); exit()
#     except Exception as e: print(f"An error occurred loading the corpus: {e}"); exit()
#
#     # --- Tokenize Corpus ---
#     print("Splitting corpus into sentences...")
#     try: all_sentences = nltk.sent_tokenize(corpus_text)
#     except LookupError: all_sentences = corpus_text.splitlines()
#     except Exception as e: print(f"An error occurred during sentence tokenization: {e}"); exit()
#     if not all_sentences: print("Error: Could not split corpus into sentences/lines."); exit()
#     print(f"Total sentences in corpus: {len(all_sentences)}")
#
#     # --- Train Language Model on FULL Corpus ---
#     print(f"\nTraining {lm_ngram_n}-gram Language Model on FULL corpus...")
#     start_time = time.time()
#     lm = Spell_Checker.Language_Model(n=lm_ngram_n, chars=False)
#     lm.build_model(corpus_text)
#     end_time = time.time()
#     print(f"Language Model training completed in {end_time - start_time:.2f} seconds.")
#     print(f"LM Vocabulary size: {len(lm.word_dict)}")
#
#     # --- Setup Spell Checker ---
#     print("\nSetting up Spell Checker...")
#     spell_checker = Spell_Checker(lm)
#     spell_checker.add_error_tables(raw_error_tables)
#     print("Spell Checker setup complete.")
#
#     # --- Select Sentences for Evaluation ---
#     num_eval_sentences = min(evaluation_set_size, len(all_sentences))
#     if num_eval_sentences < evaluation_set_size: print(f"Warning: Evaluating on {num_eval_sentences} sentences.")
#     # Use the *original* list before shuffling if you want the *exact* same sentences each time
#     # OR shuffle with the fixed seed guarantees the *same shuffled order* each time
#     random.shuffle(all_sentences) # Shuffle is now deterministic
#     evaluation_sentences = all_sentences[:num_eval_sentences] # Take first N from shuffled list
#     print(f"\nSelected {num_eval_sentences} random sentences (deterministic order) for evaluation.")
#
#     # --- Evaluation ---
#     print(f"Starting evaluation (alpha = {spell_check_alpha})...")
#     total_tested, correctly_corrected, errors_generated = 0, 0, 0
#     failed_cases_printed_summary = 0
#     max_failed_summary_prints = 10
#     first_failure_details_printed = False
#
#     start_time = time.time()
#     for i, original_sentence in enumerate(evaluation_sentences):
#         if not original_sentence.strip(): continue
#         orig_norm = normalize_text(original_sentence)
#         if not orig_norm: continue
#
#         _, erroneous_sentence = generate_sentence_with_error(
#             orig_norm, raw_error_tables, max_edits=2, prob_dist2=error_gen_prob_dist2)
#
#         if erroneous_sentence != orig_norm:
#             errors_generated += 1; total_tested += 1
#             corrected_sentence = spell_checker.spell_check(erroneous_sentence, spell_check_alpha)
#             if corrected_sentence == orig_norm:
#                 correctly_corrected += 1
#                 # Print ALL Successes
#                 print("+" * 20)
#                 print(f"Correction SUCCEEDED (Success Count: {correctly_corrected}):")
#                 print(f"  Original:  {orig_norm}")
#                 print(f"  Erroneous: {erroneous_sentence}")
#                 print(f"  Corrected: {corrected_sentence}")
#                 print("+" * 20 + "\n")
#             else:
#                 # Print limited Failed Summaries
#                 if failed_cases_printed_summary < max_failed_summary_prints:
#                     print("-" * 20); print(f"Correction FAILED (Fail Count: {failed_cases_printed_summary + 1}):")
#                     print(f"  Original:  {orig_norm}"); print(f"  Erroneous: {erroneous_sentence}")
#                     print(f"  Corrected: {corrected_sentence}"); print("-" * 20 + "\n")
#                     failed_cases_printed_summary += 1
#                 elif failed_cases_printed_summary == max_failed_summary_prints:
#                     print("... (Reached max failed cases summary to print) ...\n"); failed_cases_printed_summary += 1
#
#                 # Print Detailed Diagnostics for First Failure
#                 if not first_failure_details_printed:
#                     print("--- DETAILED DIAGNOSTICS (First Failure Only) ---")
#                     word_index_failed, original_word_failed, erroneous_word_failed = -1, "", ""
#                     err_tokens, orig_tokens = erroneous_sentence.split(), orig_norm.split()
#                     if len(err_tokens) == len(orig_tokens):
#                         for idx, (e_tok, o_tok) in enumerate(zip(err_tokens, orig_tokens)):
#                             if e_tok != o_tok:
#                                 word_index_failed, original_word_failed, erroneous_word_failed = idx, o_tok, e_tok
#                                 print(f"  Word index: {word_index_failed}")
#                                 print(f"  Erroneous word: '{erroneous_word_failed}'")
#                                 print(f"  Original word:  '{original_word_failed}'"); break
#                     # --- Include diagnostics from previous step if needed ---
#                     # ... (print candidates, scores etc. for erroneous vs original) ...
#                     print("--- END DETAILED DIAGNOSTICS ---")
#                     first_failure_details_printed = True
#
#         # Progress indicator
#         if (i + 1) % 10 == 0 or i == num_eval_sentences - 1:
#              print(f"  Processed {i + 1}/{num_eval_sentences} evaluation sentences...")
#
#     end_time = time.time()
#     print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")
#
#     # --- Results ---
#     print("\n--- Evaluation Results ---")
#     print(f"Total sentences sampled for evaluation: {num_eval_sentences}")
#     print(f"Sentences where error was generated: {errors_generated}")
#     print(f"Sentences tested with spell checker: {total_tested}")
#     if total_tested > 0:
#         accuracy = (correctly_corrected / total_tested) * 100
#         print(f"Correctly corrected sentences: {correctly_corrected}")
#         print(f"Accuracy: {accuracy:.2f}%")
#     else: print("No sentences with generated errors were tested.")


# --- Main Evaluation Block ---
if __name__ == "__main__":

    # +++ Set Random Seed for Reproducibility +++
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    # ++++++++++++++++++++++++++++++++++++++++++++

    print("Starting Spell Checker Evaluation...")
    print(f"(Using random seed: {SEED_VALUE})")

    # --- Configuration ---
    corpus_file_path = 'big.txt'
    lm_ngram_n = 3
    evaluation_set_size = 100 # Use a larger size for meaningful time measurement
    error_gen_prob_dist2 = 0.25
    spell_check_alpha = 0.7

    # --- Load Error Tables (Imported) ---
    try:
        if not isinstance(raw_error_tables, dict) or not raw_error_tables:
             print("Error: raw_error_tables is empty or not a dictionary."); exit()
        print(f"Loaded raw error tables (Number of types: {len(raw_error_tables)})")
    except NameError: print("Error: raw_error_tables not found."); exit()

    # --- Load Corpus Data ---
    try:
        with open(corpus_file_path, 'r', encoding='utf-8') as f: corpus_text = f.read()
        print(f"Loaded corpus from {corpus_file_path} ({len(corpus_text)} chars)")
    except FileNotFoundError: print(f"Error: Corpus file not found at {corpus_file_path}"); exit()
    except Exception as e: print(f"An error occurred loading the corpus: {e}"); exit()

    # --- Tokenize Corpus ---
    print("Splitting corpus into sentences...")
    try: all_sentences = nltk.sent_tokenize(corpus_text)
    except LookupError: all_sentences = corpus_text.splitlines()
    except Exception as e: print(f"An error occurred during sentence tokenization: {e}"); exit()
    if not all_sentences: print("Error: Could not split corpus."); exit()
    print(f"Total sentences in corpus: {len(all_sentences)}")

    # --- Train Language Model on FULL Corpus ---
    print(f"\nTraining {lm_ngram_n}-gram Language Model on FULL corpus...")
    lm_train_start_time = time.time() # Time LM training
    lm = Spell_Checker.Language_Model(n=3, chars=False)
    lm.build_model(corpus_text)
    lm_train_end_time = time.time()
    lm_train_duration = lm_train_end_time - lm_train_start_time # Store duration
    print(f"Language Model training completed in {lm_train_duration:.2f} seconds.")
    print(f"LM Vocabulary size: {len(lm.model_dict)}")

    # --- Setup Spell Checker ---
    print("\nSetting up Spell Checker...")
    spell_checker = Spell_Checker(lm)
    spell_checker.add_error_tables(raw_error_tables)
    print("Spell Checker setup complete.")

    # --- Select Sentences for Evaluation ---
    num_eval_sentences = min(evaluation_set_size, len(all_sentences))
    if num_eval_sentences < evaluation_set_size: print(f"Warning: Evaluating on {num_eval_sentences} sentences.")
    random.shuffle(all_sentences)
    evaluation_sentences = all_sentences[:num_eval_sentences]
    print(f"\nSelected {num_eval_sentences} random sentences for evaluation.")

    # --- Generate Evaluation Data ---
    print("Preparing evaluation data...")
    evaluation_pairs = []
    errors_generated_count = 0
    processed_for_eval = 0
    for original_sentence in evaluation_sentences:
         if not original_sentence.strip(): continue
         orig_norm = normalize_text(original_sentence)
         if not orig_norm: continue
         processed_for_eval += 1
         orig_returned, erroneous_sentence = generate_sentence_with_error(
             orig_norm, raw_error_tables, max_edits=2, prob_dist2=error_gen_prob_dist2)
         evaluation_pairs.append((orig_norm, erroneous_sentence))
         if erroneous_sentence != orig_norm: errors_generated_count += 1
    print(f"Prepared {len(evaluation_pairs)} evaluation pairs ({errors_generated_count} with generated errors).")

    # --- Evaluation ---
    print(f"\nStarting evaluation run (alpha = {spell_check_alpha})...")
    total_tested = len(evaluation_pairs)
    correctly_processed = 0

    # ---- START Timing the evaluation part ----
    eval_run_start_time = time.time()

    for i, (orig_norm, input_sentence) in enumerate(evaluation_pairs):
        corrected_sentence = spell_checker.spell_check(input_sentence, spell_check_alpha)
        # if corrected_sentence != orig_norm:
        #     print("original sentence: ", orig_norm)
        #     print("the errored sentence: ", input_sentence)
        #     print("the model correction: ", corrected_sentence)
        if corrected_sentence == orig_norm:
            correctly_processed += 1
        # --- NO PRINTS INSIDE THIS LOOP FOR PERFORMANCE ---

    # ---- END Timing the evaluation part ----
    eval_run_end_time = time.time()
    eval_duration = eval_run_end_time - eval_run_start_time # Store duration
    print(f"Evaluation run finished processing {total_tested} sentences in {eval_duration:.2f} seconds.")

    # --- Results ---
    print("\n--- Evaluation Results ---")
    print(f"Total sentences sampled: {num_eval_sentences}")
    print(f"Total non-empty normalized sentences prepared: {processed_for_eval}")
    print(f"Sentences where distinct error was generated: {errors_generated_count}")
    print(f"Total sentences tested with spell checker: {total_tested}")
    if total_tested > 0:
        accuracy = (correctly_processed / total_tested) * 100
        print(f"Sentences processed correctly (fixed or left unchanged): {correctly_processed}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        if eval_duration > 0:
             sentences_per_sec = total_tested / eval_duration
             print(f"Processing speed: {sentences_per_sec:.2f} sentences/sec")
    else:
        print("No sentences were processed for evaluation.")

    # --- TOTAL TIME CALCULATION AND PRINT ---
    total_processing_time = lm_train_duration + eval_duration
    print("-" * 20)
    print(f"Total Time (LM Training + Evaluation Run): {total_processing_time:.2f} seconds")
    print("-" * 20)
