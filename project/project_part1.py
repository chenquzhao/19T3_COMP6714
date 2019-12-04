## Import Libraries and Modules here...
import spacy
import math
from itertools import permutations

class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = {}
        self.tf_entities = {}

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = {}
        self.idf_entities = {}

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        nlp = spacy.load("en_core_web_sm")
        doc_num = 0
        tf_tokens, tf_entities, idf_tokens, idf_entities = {}, {}, {}, {}

        for document in documents.values():
            doc = nlp(document)
            doc_num += 1
            count, count2 = {}, {}

            for token in doc:
                if token.is_stop or token.is_punct:
                    continue

                if token.text in count:
                    count[token.text] += 1
                else:
                    count[token.text] = 1

            for ent in doc.ents:
                if ent.text in count2:
                    count2[ent.text] += 1
                else:
                    count2[ent.text] = 1

                if len(ent.text.split()) == 1:
                    if ent.text in count:
                        if count[ent.text] == 1:
                            del count[ent.text]
                        else:
                            count[ent.text] -= 1

            for key in count.keys():
                if key in tf_tokens:
                    tf_tokens[key] = {**tf_tokens[key],
                                      **{doc_num: 1 + math.log(1 + math.log(count[key]))}}
                else:
                    tf_tokens[key] = {doc_num: 1 + math.log(1 + math.log(count[key]))}

            for key in count2.keys():
                if key in tf_entities:
                    tf_entities[key] = {**tf_entities[key], **{doc_num: 1 + math.log(count2[key])}}
                else:
                    tf_entities[key] = {doc_num: 1 + math.log(count2[key])}

        for key in tf_tokens.keys():
            idf_tokens[key] = 1 + math.log(doc_num / (1 + len(tf_tokens[key])))

        for key in tf_entities.keys():
            idf_entities[key] = 1 + math.log(doc_num / (1 + len(tf_entities[key])))

        self.tf_tokens = dict(tf_tokens)
        self.tf_entities = dict(tf_entities)

        self.idf_tokens = dict(idf_tokens)
        self.idf_entities = dict(idf_entities)
        
    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        id_list, split_re = [], []

        for i in range(len(DoE) + 1):

            for j in permutations(DoE.keys(), i):
                token_lis = Q.split()
                tup = self.handle_token(token_lis, list(j), DoE)

                if tup[2] not in id_list:
                    id_list.append(tup[2])
                    sp_item = {'tokens': tup[0], 'entities': tup[1]}
                    split_re.append(sp_item)

        return split_re

    def handle_token(self, token_list, entity_list, DoE):
        id_com = []

        for i in range(len(entity_list)):
            split = entity_list[i].split()
            mark = 0
            token_mark = token_list.copy()

            for j in range(len(split)):
                try:
                    if token_list.index(split[j]) < mark:
                        token_list = token_mark.copy()
                        break
                    else:
                        mark = token_list.index(split[j])
                        del token_list[mark]
                except ValueError:
                    token_list = token_mark.copy()
                    break

            if token_mark != token_list:
                # entity_co.append(entity_list[i])
                id_com.append(DoE[entity_list[i]])
                id_com.sort()

        tup = (token_list, entity_list, id_com)
        return tup

    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        max_score = 0
        max_sp = {}

        for i in range(len(query_splits)):
            token_sc, entity_sc = 0, 0

            for key in query_splits[i].keys():
                if key == 'tokens':
                    for j in range(len(query_splits[i][key])):
                        w = query_splits[i][key][j]
                        if w in self.tf_tokens.keys():
                            if doc_id in self.tf_tokens[w].keys():
                                token_sc += self.tf_tokens[w][doc_id] * self.idf_tokens[w]

                if key == 'entities':
                    for j in range(len(query_splits[i][key])):
                        y = query_splits[i][key][j]
                        if y in self.tf_entities.keys():
                            if doc_id in self.tf_entities[y].keys():
                                entity_sc += self.tf_entities[y][doc_id] * self.idf_entities[y]

            score = 0.4 * token_sc + entity_sc

            if score > max_score:
                max_score = score
                max_sp.update(query_splits[i])

        tup = (max_score, max_sp)
        return tup
