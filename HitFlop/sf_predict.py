import tqdm
import sqlite3
from itertools import (takewhile, repeat)

__author__ = 'rossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


class PredictAndEvaluate(object):

    def __init__(self, database_name, test_adoption_log):
        self.database_name = database_name
        self.test_adoption_log = test_adoption_log
        self.thresholds, self.adopters = self.__get_model()
        self.__build_db()

    @staticmethod
    def __rawincount(filename):
        f = open(filename, 'rb')
        buf_gen = takewhile(lambda x: x, (f.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in buf_gen)

    @staticmethod
    def get_hit_flop(hit_flop_map):
        """
        Configure the hit/flop training split

        :param hit_flop_map:
        :return:
        """
        f = open(hit_flop_map)
        hits_test, flops_test = [], []
        for l in f:
            l = l.rstrip().split(",")
            if int(l[1]) == -1:
                flops_test.append(l[0])
            else:
                hits_test.append(l[0])
        return hits_test, flops_test

    def __build_db(self):
        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS adoptions_test;""")
        conn.execute("""CREATE TABLE  adoptions_test
                       (good TEXT  NOT NULL,
                       adopter TEXT NOT NULL,
                       slot      INTEGER NOT NULL,
                       quantity  INTEGER
                       );""")

        conn.close()

    def load_test_set(self):
        conn = sqlite3.connect("%s" % self.database_name)
        f = open(self.test_adoption_log)
        f.next()
        count = 0
        total = self.__rawincount(self.test_adoption_log)

        for row in tqdm.tqdm(f, total=total):
            row = row.rstrip().split(",")
            conn.execute("""INSERT into adoptions_test (good, adopter, slot, quantity) VALUES ('%s', '%s', %d, %d)""" %
                         (row[0], row[1], int(row[2]), int(row[3])))
            count += 1
            if count % 10000 == 0:
                conn.commit()
        conn.execute('CREATE INDEX IF NOT EXISTS good_test_idx on adoptions_test(good)')
        conn.execute('CREATE INDEX IF NOT EXISTS adopter_test_idx on adoptions_test(adopter)')
        conn.execute('CREATE INDEX IF NOT EXISTS slot_test_idx on adoptions_test(slot)')
        conn.close()

    def __get_model(self):
        conn = sqlite3.connect("%s" % self.database_name)
        curr = conn.cursor()
        curr.execute("""SELECT * from model""")
        models = curr.fetchall()

        thresholds = {}
        adopters = {}
        for m in models:
            thresholds[m[0]] = m[2]
            curr.execute("SELECT adopter from %s" % m[1])
            ads = curr.fetchall()
            adopters[m[0]] = {a[0]: None for a in ads}

        print adopters

        curr.close()
        conn.close()

        return thresholds, adopters

    def predict(self,  slots):

        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS predictions;""")
        conn.execute("""CREATE TABLE predictions
            (good TEXT  NOT NULL,
            success INTEGER NOT NULL DEFAULT 0
            );""")

        cur = conn.cursor()
        cur.execute("""SELECT distinct good from adoptions_test""")
        goods = cur.fetchall()

        predictions = {}
        conn = sqlite3.connect("%s" % self.database_name)
        curr = conn.cursor()

        for good in goods:
            curr.execute("""SELECT * from adoptions_test where good='%s' and slot<=%d order by slot asc""" % (good[0], int(slots)))
            good_adoptions = curr.fetchall()
            for ad in good_adoptions:

                if ad[0] not in predictions:
                     predictions[ad[0]] = {}
                     predictions[ad[0]]["numero_adozioni_tot"] = 0
                     predictions[ad[0]]["adozioni_innovatori"] = 0
                     predictions[ad[0]]["adozioni_flop"] = 0
                if ad[1] in self.adopters['hit']:  #
                     predictions[ad[0]]["adozioni_innovatori"] += 1
                elif ad[1] in self.adopters['flop']:
                     predictions[ad[0]]["adozioni_flop"] += 1
                predictions[ad[0]]["numero_adozioni_tot"] += 1

        count = 0
        for g in predictions:
            if predictions[g]["adozioni_innovatori"] > 0 or predictions[g]["adozioni_flop"] > 0:
                print predictions[g], count
                count += 1

        for good in predictions:
            # print predictions[good]
            if float(predictions[good]["numero_adozioni_tot"]) >= 0:
                percentage_hitters = float("{0:.3f}".format(float(predictions[good]["adozioni_innovatori"]) / \
                                        float(predictions[good]["numero_adozioni_tot"])))
                percentage_floppers = float("{0:.3f}".format(float(predictions[good]["adozioni_flop"]) / \
                                          float(predictions[good]["numero_adozioni_tot"])))

                if percentage_hitters >= self.thresholds['hit'] and percentage_floppers < self.thresholds['flop']:
                     conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                elif percentage_floppers >= self.thresholds['flop'] and percentage_hitters < self.thresholds['hit']:
                     conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                elif percentage_floppers >= self.thresholds['flop'] and percentage_hitters >= self.thresholds['hit']:
                    if self.thresholds['flop'] > 0 and self.thresholds['hit'] > 0:
                        dist_flop = percentage_floppers - self.thresholds['flop']
                        dist_hit = percentage_hitters - self.thresholds['hit']
                        if dist_flop >= dist_hit:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                        else:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                    elif self.thresholds['flop'] == 0 and self.thresholds['hit'] == 0:
                        if percentage_floppers > percentage_hitters:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                        elif percentage_floppers < percentage_hitters:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                        else:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % good)
                    else:
                        if (self.thresholds['flop'] == 0 and
                                percentage_floppers >= (percentage_hitters-self.thresholds['hit'])) or \
                                (self.thresholds['hit'] == 0 and percentage_hitters < (percentage_floppers-self.thresholds['flop'])):
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                        elif (self.thresholds['flop'] == 0 and percentage_floppers < (percentage_hitters-self.thresholds['hit'])) or (self.thresholds['hit'] == 0 and percentage_hitters >= (percentage_floppers - self.thresholds['flop'])):
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                        else:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % good)
                else:
                    conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % good)

        conn.commit()
        conn.close()

    def evaluate(self, ground_truth_file):

        gt_hits, gt_flops = self.get_hit_flop(ground_truth_file)

        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT * from predictions""")
        predictions = cur.fetchall()

        TP, FP, FN, TN = 0, 0, 0, 0
        unclassified = 0
        total = 0

        for res in predictions:

            total += 1
            if int(res[1]) == 0:
                unclassified += 1
            elif int(res[1]) == 1:
                if res[0] in gt_hits:
                    TP += 1
                else:
                    FP += 1
            else:
                if res[0] in gt_flops:
                    TN += 1
                else:
                    FN += 1

        if TP+FP != 0:
            precision = float(TP)/float(TP+FP)
        else:
            precision = 0.0

        if TP+FN+unclassified != 0:
            frecall = float(TP)/float(TP+FN+unclassified)
        else:
            frecall = 0.0

        if float(total) != 0:
            accuracy = float(TP+TN)/float(total)
        else:
            accuracy = 0.0

        if FN+TN != 0:
            NPV = float(TN)/float(FN+TN)
        else:
            NPV = 0.0

        if TP+FN != 0:
            recall = float(TP)/float(TP+FN)
        else:
            recall = 0.0

        if FP+TN != 0:
            specificity = float(TN)/float(FP+TN)
        else:
            specificity = 0.0

        if FP+TN != 0:
            un_specificity = float(TN)/float(FP+TN+unclassified)
        else:
            un_specificity = 0.0

        if TP+FP+FN != 0:
            F1 = 2*float(TP)/float(2*TP+FP+FN)
        else:
            F1 = 0

        conn.execute("""DROP TABLE IF EXISTS prediction_results;""")
        conn.execute("""CREATE TABLE prediction_results (score TEXT NOT NULL, value REAL NOT NULL);""")
        conn.commit()

        conn.execute("""INSERT INTO prediction_results (score, value) VALUES ('precision', %f)""" % precision)
        conn.execute("""INSERT INTO prediction_results (score, value) VALUES ('recall (with unclassified)', %f)""" % frecall)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('recall (without unclassified)', %f)""" % recall)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('accuracy', %f)""" % accuracy)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('specificity (with unclassified)', %f)""" % un_specificity)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('specificity (with unclassified)', %f)""" % specificity)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('NPV', %f)""" % NPV)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('F1', %f)""" % F1)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('TP', %d)""" % TP)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('FP', %d)""" % FP)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('TN', %d)""" % TN)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('FN', %d)""" % FN)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('unclassified', %d)""" % unclassified)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('total', %d)""" % total)

        conn.commit()
        conn.close()

if __name__ == "__main__":
    import argparse

    print "------------------------------------"
    print "          Superforecasters          "
    print "------------------------------------"
    print "Author: ", __author__
    print "Email:  ", __email__
    print "------------------------------------"

    parser = argparse.ArgumentParser()

    parser.add_argument('db', type=str, help='desired db name', default="results.db")
    parser.add_argument('log', type=str, help='test adoption_log file')
    parser.add_argument('gt', type=str, help='training set ground_truth file')
    parser.add_argument('slots', type=int, help='observed slots', default=4)

    args = parser.parse_args()

    # Load Adoption log Data
    print "Loading data"
    l = PredictAndEvaluate(args.db, args.log)
    l.load_test_set()

    print "Predict"
    l.predict(args.slots)

    print "Evaluate"
    l.evaluate(args.gt)
