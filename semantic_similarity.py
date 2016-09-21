# usado para importar o csv
import csv
# Usado para fazer o tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
# diretorio dataset
directory_dataset = 'data_academia.csv'

def calc_tfidf():
    with open('data_academia_results.csv', 'w') as csvfile:
        # fields do csv
        fieldnames = ['id_question',
                      'answer_count',
                      'AcceptedAnswerId',
                      'body_question',
                      'id_author_question',
                      'name_author_question',
                      'reputation_author_question',
                      'id_answer',
                      'body_answer',
                      'id_author_answer',
                      'name_author_answer',
                      'reputation_author_answer',
                      'score_answer',
                      'id_parent',
                      'accepted_answer',
                      'tf_idf']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with open(directory_dataset) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                vect = TfidfVectorizer(min_df=1)
                list = [row['body_question'], row['body_answer']]
                tfidf = vect.fit_transform(list)
                result = (tfidf * tfidf.T).A
                writer.writerow({'id_question': row['id_question'],
                                 'answer_count': row['answer_count'],
                                 'AcceptedAnswerId': row['AcceptedAnswerId'],
                                 'body_question': row['body_question'],
                                 'id_author_question': row['id_author_question'],
                                 'name_author_question': row['name_author_question'],
                                 'reputation_author_question': row['reputation_author_question'],
                                 'id_answer': row['id_answer'],
                                 'body_answer': row['body_answer'],
                                 'id_author_answer': row['id_author_answer'],
                                 'name_author_answer': row['name_author_answer'],
                                 'reputation_author_answer': row['reputation_author_answer'],
                                 'score_answer': row['score_answer'],
                                 'id_parent': row['id_parent'],
                                 'accepted_answer': row['accepted_answer'],
                                 'tf_idf': result[0][1]})

calc_tfidf()

