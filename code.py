import re
import nltk
import math
import random
import collections

nltk.download('gutenberg')


class Spell_Checker:
    """
        The class implements a context-sensitive spell checker. The corrections
            are done in the Noisy Channel framework, based on a language model and
            an error distribution model.
    """

    def __init__(self, lm=None):
        """
            Initializing a spell checker object with a language model as an
                instance  variable.

                Args:
                    lm: a language model object. Defaults to None.
        """

        self.lm = lm
        self.error_tables = None
        self.edit1_candidates = {}
        self.edit2_candidates = {}
        self.all_candidates = {}

    def add_language_model(self, lm=None):
        """
            Adds the specified language model as an instance variable.
                (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """

        self.lm = lm

    def add_error_tables(self, error_tables):
        """
            Adds the specified dictionary of error tables as an instance variable.
                (Replaces an older value dictionary if set)

            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0

        """

        probabilities = {'insertion': {}, 'deletion': {}, 'substitution': {}, 'transposition': {}}

        for error_type, table in error_tables.items():
            total = sum(table.values()) + 1e-10  # avoid division by zero
            for pair, count in table.items():
                probabilities[error_type][pair] = count / total
        self.error_tables = probabilities

    def evaluate_text(self, text):
        """
            Returns the log-likelihood of the specified text given the language
                model in use. Smoothing should be applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """

        if self.lm is None:
            return 0.0
        else:
            return self.lm.evaluate_text(text)

    def edits1(self, word):
        """

        :param word: str, a word  to generate edits from
        :return: All edits that are one edit away from `word`.
        """

        if word in self.edit1_candidates:
            return self.edit1_candidates[word]

        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [(L + R[1:], ("deletion", R[0])) for L, R in splits if R]
        transposes = [(L + R[1] + R[0] + R[2:], ("transposition", R[:2])) for L, R in splits if len(R) > 1]
        replaces = [(L + c + R[1:], ("substitution", (R[0], c))) for L, R in splits if R for c in letters]
        inserts = [(L + c + R, ("insertion", c)) for L, R in splits for c in letters]

        all_edits = deletes + transposes + replaces + inserts

        # Filter only known candidates
        known_edits = [(w, info) for (w, info) in all_edits if w in self.lm.word_dict]

        result = set(known_edits)
        self.edit1_candidates[word] = result
        return result

    def edits2(self, word, edits1=None):
        """

        :param word: str, a word  to generate edits from
        :param edits1: list, edits that are 1 edit away from the given word
        :return: All edits that are two edits away from `word`.
        """
        if word in self.edit2_candidates:
            return self.edit2_candidates[word]

        if edits1 is None:
            edits1 = self.edits1(word)

        known_words = set()
        for (e1, info1) in edits1:
            for (e2, info2) in self.edits1(e1):
                if e2 != word and e2 in self.lm.word_dict:
                    known_words.add((e2, info1 + info2))

        self.edit2_candidates[word] = known_words
        return known_words

    def known(self, words):
        """
        Filter and return only the known 1-gram tokens from the given word list.

        :param words: list of 1-gram tokens to check
        :return: list of tokens recognized by the language model
        """
        vocab = self.lm.word_dict
        return [(w, info) for w, info in words if w in vocab]

    def candidates(self, word):
        """
        Generate all candidates for a given word (1-gram token) within an edit distance of 0 to 2.

        Args:
            word (str): The original word to generate edits from.

        Returns:
            edits (set of tuples): Each tuple contains:
                - edited_word (str): A candidate word within edit distance 0, 1, or 2.
                - error_info_tuple:
                    * For edit distance 0: None
                    * For edit distance 1: (error_type, error)
                    * For edit distance 2: (error_type1, error1, error_type2, error2)
                (This tuple stores the operations used to generate the candidate for efficiency.)
        """

        if word in self.all_candidates:
            return self.all_candidates[word]

        # Edit distance 0 — word is known
        edit0 = self.known([(word, None)])

        # Edit distance 1
        edit1_raw = self.edits1(word)
        edit1 = self.known(edit1_raw)

        # Edit distance 2
        edit2_raw = self.edits2(word, edit1_raw)
        edit2 = self.known(edit2_raw)

        all_candidates = set(edit0 + edit1 + edit2)
        self.all_candidates[word] = all_candidates
        return all_candidates

    def spell_check(self, text, alpha):
        """
        Returns the most probable fix for the specified text using a noisy channel model.

        Args:
            text (str): the text to spell check.
            alpha (float): the probability of keeping a lexical word as is.

        Returns:
            A corrected string (or a copy of the original if no corrections are made.)
        """
        if self.lm is None or self.error_tables is None:
            return text

        processed = normalize_text(text)
        tokens = [tok[0] for tok in get_ngrams(processed, 1)]
        size = len(tokens)
        corrected = tokens.copy()
        changed = False

        for i, token in enumerate(tokens):
            if self.lm.known(token):
                continue

            candidates = self.candidates(token)
            final_candidates = []

            for cand_word, err_info in candidates:
                test_tokens = corrected.copy()
                test_tokens[i] = cand_word
                candidate_sentence = ' '.join(test_tokens)
                lang_score = self.evaluate_text(candidate_sentence)

                if err_info is None:
                    error_prob = alpha
                else:
                    error_prob = 1 - alpha
                    if len(err_info) == 2:
                        error_type, error = err_info
                        error_prob *= self.error_tables[error_type].get(error, 1e-10)
                    elif len(err_info) == 4:
                        t1_, e1, t2_, e2 = err_info
                        error_prob *= (
                                self.error_tables[t1_].get(e1, 1e-10) *
                                self.error_tables[t2_].get(e2, 1e-10)
                        )

                total_score = lang_score * error_prob
                final_candidates.append((cand_word, total_score))
            if final_candidates:
                final_candidates.sort(key=lambda x: x[1], reverse=True)
                final_candidates_2 = [x for x in final_candidates if x[1] != -0.0 and x[1] != float('-inf')]
                if not final_candidates_2:
                    best_cand, best_score = final_candidates[0]
                else:
                    best_cand, best_score = max(final_candidates_2, key=lambda x: x[1])
                corrected[i] = best_cand
                changed = True

        if not changed:
            best_score = self.evaluate_text(' '.join(tokens))
            best_version = tokens.copy()

            for i in range(size):
                original_word = tokens[i]
                candidates = self.candidates(original_word)

                for cand_word, err_info in candidates:
                    if cand_word == original_word:
                        continue

                    test_tokens = tokens.copy()
                    test_tokens[i] = cand_word
                    sentence = ' '.join(test_tokens)
                    score = self.evaluate_text(sentence)

                    if score > best_score:
                        best_score = score
                        best_version = test_tokens

            corrected = best_version

        return ' '.join(corrected)

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """
            The class implements a Markov Language Model that learns a model from a given text.
                It supports language generation and the evaluation of a given string.
                The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """
                Initializing a language model object.
                Args:
                    n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                    chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                                  Defaults to False
            """

            self.n = n
            self.chars = chars
            self.model_dict = None
            self.word_dict = None
            self.token_frequencies = None
            self.ordered_tokens_by_frequency = None
            self.vocab_size = None
            self.total_occurrences = None

            if chars:
                self.separator = ''
            else:
                self.separator = ' '

        def get_model_dictionary(self):
            """
            Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """
            Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def known(self, word):
            """
            Checks if the given word (1-gram token) is known by the language model.

            :param word: str, a single word token.
            :return: bool, True if the word exists in the language model's vocabulary.
            """
            return word in self.word_dict

        def build_model(self, text):
            """
            populates the instance variable model_dict.


                Args:
                    text (str): the text to construct the model from.
            """
            text = normalize_text(text)
            self.model_dict = collections.Counter(get_ngrams(text, self.n, self.chars))
            self.word_dict = set(collections.Counter(get_ngrams(text, 1)).keys())
            self.word_dict = set([i[0] for i in self.word_dict])

            # Create a list of token frequencies ordered by most common tokens
            self.token_frequencies = []
            self.ordered_tokens_by_frequency = self.model_dict.most_common()

            self.total_occurrences = sum(self.model_dict.values())
            self.vocab_size = len(self.model_dict)

            for token, frequency in self.ordered_tokens_by_frequency:
                self.token_frequencies.append(frequency)

        def sample_weighted_ngram(self, prefix_filter=None, min_frequency=1):
            """
            Samples a random n-gram from the language model based on frequency distribution and optional filters.

            Args:
                prefix_filter (tuple or None): A prefix to filter n-grams (length ≤ n). If None, samples from all n-grams.
                min_frequency (int): Minimum count an n-gram must have to be considered.

            Returns:
                tuple: (ngram, count) - the sampled n-gram and its frequency.
                       Returns (None, None) if no matching n-gram is found.
            """

            # Sample from full distribution if no prefix is provided
            if prefix_filter is None:
                return random.choices(self.ordered_tokens_by_frequency, weights=self.token_frequencies)[0]

            # Otherwise, filter n-grams based on prefix and frequency
            prefix_length = len(prefix_filter)
            filtered_ngrams = [
                (ngram, freq) for (ngram, freq) in self.ordered_tokens_by_frequency
                if freq >= min_frequency and ngram[:prefix_length] == prefix_filter
            ]

            if not filtered_ngrams:
                return None, None

            weights = [freq for (_, freq) in filtered_ngrams]
            return random.choices(filtered_ngrams, weights=weights)[0]

        def generate(self, context=None, n=20, min_ngram_count=2):
            """
            Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the nth word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.


                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.
                    min_ngram_count (int): a minimum count threshold that a ngram has to have in order to be sampled

                Return:
                    String. The generated text.

            """

            if self.model_dict is None:
                return ""
                # return context

            if context is None:
                # Sample a random starting n-gram if no context is provided
                sampled_ngram, _ = self.sample_weighted_ngram(min_frequency=min_ngram_count)
                generated_tokens = list(sampled_ngram)
            else:
                normalized_text = normalize_text(context)
                tokenized_context = get_ngrams(normalized_text, 1, self.chars)
                generated_tokens = [token[0] for token in tokenized_context]

            # Case 1: context already satisfies desired length
            if len(generated_tokens) >= n:
                return self.separator.join(generated_tokens[:n])

            # Case 2: context too short to form a full (n-1)-gram prefix
            elif len(generated_tokens) < self.n:
                prefix = tuple(generated_tokens)
                sampled_ngram, _ = self.sample_weighted_ngram(prefix, min_frequency=min_ngram_count)

                if sampled_ngram is None:
                    return context
                generated_tokens = list(sampled_ngram)

            # Generate more tokens until max_length is reached
            current_index = len(generated_tokens) - (self.n - 1)
            while len(generated_tokens) < n:
                prefix_ngram = tuple(generated_tokens[current_index:current_index + self.n - 1])
                sampled_ngram, _ = self.sample_weighted_ngram(prefix_ngram, min_frequency=min_ngram_count)

                if sampled_ngram is None:
                    break

                next_token = sampled_ngram[-1]
                generated_tokens.append(next_token)
                current_index += 1

            return self.separator.join(generated_tokens)

        def evaluate_text(self, text):
            """
                Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing should be applied if necessary.


               Args:
                   text (str): Text to evaluate.


               Returns:
                   Float. The float should reflect the (log) probability.
            """

            normalized_text = normalize_text(text)
            tokenized_context = get_ngrams(normalized_text, self.n, self.chars)

            if not tokenized_context:
                return float("-inf")

            likelihood = 0.0

            for ngram in tokenized_context:
                ngram_str = self.separator.join(ngram)
                prob = self.smooth(ngram_str)

                if prob > 0:
                    likelihood += math.log(prob, 2)
                else:
                    likelihood += math.log(1 / (self.total_occurrences + self.vocab_size), 2)

            # return likelihood
            return likelihood / len(tokenized_context)

        def smooth(self, ngram):
            """
            Returns the smoothed (Laplace) probability of the specified ngram.


                Args:
                    ngram (str): the ngram to have its probability smoothed


                Returns:
                    float. The smoothed probability.
            """
            if self.model_dict is None:
                return float("-inf")

            if self.chars:
                ngram_tuple = tuple(ngram)
            else:
                ngram_tuple = tuple(ngram.split())

            if len(ngram_tuple) != self.n:
                return float("-inf")

            count = self.model_dict.get(ngram_tuple, 0)
            smoothed_prob = (count + 1) / (self.total_occurrences + self.vocab_size)

            return smoothed_prob


def normalize_text(text):
    """
        Normalize input text: lowercase, endings, remove punctuation, and trim spaces.
        Returns: string. the normalized text.
        Parameters:
                text (str): input text
    """

    # Lowercase
    text = text.lower()

    # Remove tags like <s>
    text = re.sub(r'<[a-zA-Z0-9]+>', '', text)

    # Remove punctuation (keep spaces, alphabetic characters, and numbers)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Replace multiple spaces with one
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text


def get_ngrams(text, n=3, chars=False):
    """
        Convert text to a list of n-grams.

        Parameters:
            text (str): preprocessed input text
            n (int): n-gram size
            chars (bool): if True, generate character-level n-grams

        Returns:
            list of n-gram tuples
    """
    if chars:
        tokens = list(text.replace('\n', ' ').strip())
    else:
        tokens = text.split()

    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
