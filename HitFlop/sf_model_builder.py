import networkx as nx
import math
import sqlite3
import numpy as np
import operator
import tqdm
import sys
from itertools import (takewhile, repeat)

__author__ = 'rossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


class LoadData(object):

    def __init__(self, adoption_log, database_name):
        self.adoption_log = adoption_log
        self.database_name = database_name
        self.__build_db()

    @staticmethod
    def __rawincount(filename):
        f = open(filename, 'rb')
        buf_gen = takewhile(lambda x: x, (f.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in buf_gen)

    def __build_db(self):
        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""CREATE TABLE IF NOT EXISTS adoptions
                       (good TEXT  NOT NULL,
                       adopter TEXT NOT NULL,
                       slot      INTEGER NOT NULL,
                       quantity  INTEGER
                       );""")

        conn.close()

    def load(self):
        conn = sqlite3.connect("%s" % self.database_name)
        f = open(self.adoption_log)
        f.next()
        count = 0
        total = self.__rawincount(self.adoption_log)
        for row in tqdm.tqdm(f, total=total):
            row = row.rstrip().split(",")
            conn.execute("""INSERT into adoptions (good, adopter, slot, quantity) VALUES ('%s', '%s', %d, %d)""" %
                         (row[0], row[1], int(row[2]), int(row[3])))
            count += 1
            if count % 10000 == 0:
                conn.commit()
        conn.execute('CREATE INDEX good_idx on adoptions(good)')
        conn.execute('CREATE INDEX adopter_idx on adoptions(adopter)')
        conn.execute('CREATE INDEX slot_idx on adoptions(slot)')
        conn.close()

    @staticmethod
    def set_hit_flop(hit_flop_map):
        """
        Configure the hit/flop training split

        :param hit_flop_map:
        :return:
        """
        f = open(hit_flop_map)
        hits_train, flops_train = [], []
        for l in f:
            l = l.rstrip().split(",")
            if int(l[1]) == -1:
                flops_train.append(l[0])
            else:
                hits_train.append(l[0])
        return hits_train, flops_train


class EarlyAdoptersThreshold(object):

    def __init__(self, database_name):
        self.database_name = database_name

    def __identify_max(self, row, index):
        if float(row[index]) <= float(row[index + 1]) and index < len(row)-2:
            return self.__identify_max(row, index + 1)
        else:
            return index

    @staticmethod
    def __compute_distance(x1, y1, x2, y2, x3, y3):
        y4 = (y2-y1)*(x3-x1)/(x2-x1)+y1
        return math.fabs(y4-y3)

    def execute(self):
        """
        Identify the final slot of innovators adoptions.

        :param goods_adoption_trends: dictionary {good_id: [n_adoptions_0,..., n_adoptions_N]}
        :return: dictionary {good_id: id_slot}
        """

        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT distinct good from adoptions""")
        goods = cur.fetchall()

        goods_thresholds = {}
        for good in goods:
            good = good[0]
            distance_max = 0.0
            slot_threshold = 0

            cur = conn.cursor()
            cur.execute("""SELECT slot, count(*) from adoptions where good='%s' group by slot order by slot asc""" % good)
            res = cur.fetchall()
            adoption_trend = []
            idx = 1
            for x in res:
                if idx == x[0]:
                    adoption_trend.append(x[1])
                    idx += 1
                else:
                    while idx < x[0]:
                        adoption_trend.append(0)
                        idx += 1

            # adoption_trend = goods_adoption_trends[good]
            for slot in range(0, len(adoption_trend)-1):
                if float(adoption_trend[slot]) <= float(adoption_trend[slot + 1]):
                    if slot < len(adoption_trend)-2:
                        indice_max = self.__identify_max(adoption_trend, slot + 1)
                    else:
                        indice_max = len(adoption_trend)-1
                    if slot != 1:
                        tmp_distance = self.__compute_distance(slot - 1, float(adoption_trend[slot - 1]), indice_max,
                                            float(adoption_trend[indice_max]), slot, float(adoption_trend[slot]))
                    else:
                        tmp_distance = self.__compute_distance(slot, float(adoption_trend[slot]), indice_max,
                                            float(adoption_trend[indice_max]), slot+1, float(adoption_trend[slot+1]))

                    if tmp_distance > distance_max:
                        distance_max = tmp_distance
                        slot_threshold = slot

            goods_thresholds[good] = slot_threshold
        return goods_thresholds
    

class HFPropensity(object):

    def __init__(self, database_name, hits_train, flops_train, goods_to_threshold):
        self.flops_train = flops_train
        self.hits_train = hits_train
        self.goods_to_threshold = goods_to_threshold
        self.database_name = database_name
        self.__build_table()

    def __build_table(self):
        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute('DROP TABLE if EXISTS HFscore')
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS HFscore
                               (adopter TEXT NOT NULL,
                               value      REAL NOT NULL,
                               hits INTEGER NOT NULL DEFAULT 0,
                               flops INTEGER NOT NULL DEFAULT 0
                               );""")

        conn.execute('DROP TABLE if EXISTS Coverage')
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS Coverage
                                       (adopter TEXT NOT NULL,
                                       good      TEXT NOT NULL,
                                       hit INTEGER NOT NULL DEFAULT 0
                                       );""")

        conn.close()

    def execute_old(self):

        adopters_to_hits = self.__build_adopter_to_innovator(self.hits_train, self.goods_to_threshold)
        conn = sqlite3.connect("%s" % self.database_name)

        adopters_to_flops = {}
        for flop in self.flops_train:
            cur = conn.cursor()
            cur.execute("""SELECT * from adoptions where good='%s';""" % flop)
            adoptions = cur.fetchall()
            for a in adoptions:
                if a[1] not in adopters_to_hits:
                    adopters_to_hits[a[1]] = {}
                    adopters_to_hits[a[1]]["innovator"] = 0
                    adopters_to_hits[a[1]]["notInnovator"] = 0
                    adopters_to_hits[a[1]]["insuccesso"] = 0
                if a[1] not in adopters_to_flops:
                    adopters_to_flops[a[1]] = {}
                    adopters_to_flops[a[1]]['insuccesso'] = 1
                    adopters_to_flops[a[1]]['goods'] = {flop: 0}
                else:
                    if flop not in adopters_to_flops[a[1]]['goods']:
                        adopters_to_flops[a[1]]['insuccesso'] += 1
                        adopters_to_hits[a[1]]["insuccesso"] += 1
                        adopters_to_flops[a[1]]['goods'][flop] = 0

        adopter_to_epsilon_mu = self.__compute_hf_propensity(adopters_to_hits, adopters_to_flops)

        for a in adopter_to_epsilon_mu:
            hits, flops = 0, 0
            if a in adopters_to_hits and 'goods' in adopters_to_hits[a]:
                hits = len(adopters_to_hits[a]['goods'])
            if a in adopters_to_flops and 'goods' in adopters_to_flops[a]:
                flops = len(adopters_to_flops[a]['goods'])

            conn.execute("""INSERT into HFscore (adopter, value, hits, flops) VALUES ('%s', %f, %d, %d)""" %
                            (a, adopter_to_epsilon_mu[a]["misura"], hits, flops))
        conn.commit()

        for a in tqdm.tqdm(adopters_to_flops):
            if adopter_to_epsilon_mu[a]["misura"] < 0:
                if 'goods' in adopters_to_flops[a]:
                    for g in adopters_to_flops[a]['goods']:
                        conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 0)""" %
                                     (a, g))
        conn.commit()

        for a in tqdm.tqdm(adopters_to_hits):
            if adopter_to_epsilon_mu[a]["misura"] > 0:
                if 'goods' in adopters_to_hits[a]:
                    for g in adopters_to_hits[a]['goods']:
                        conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 1)""" %
                                     (a, g))
        conn.commit()

        conn.close()

    def execute(self):
        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT distinct(adopter) from adoptions;""")
        adopters = cur.fetchall()

        adopters_to_hit = {}
        adopters_to_flop = {}
        for ad in adopters:
            cur.execute("""SELECT good, min(slot) as start from adoptions where adopter=%s group by good order by min(slot) asc;""" % ad)
            first_adoptions = cur.fetchall()
            scores = {}
            f = 0
            hi = 0
            hl = 0

            if ad not in adopters_to_hit:
                adopters_to_hit[ad] = []
            if ad not in adopters_to_flop:
                adopters_to_flop[ad] = []

            last_it = None

            for i in first_adoptions:

                if last_it is None:
                    last_it = i[1]

                n = i[1]
                if n > last_it + 1:
                    last_it += 1
                    while last_it < n:
                        scores[last_it] = (0, scores[last_it-1][0])
                        last_it += 1

                last_it = n

                if i[0] in self.flops_train:
                    adopters_to_flop[ad].append(i[0])
                    f -= 1
                else:
                    if i[1] <= self.goods_to_threshold[i[0]]:
                        adopters_to_hit[ad].append(i[0])
                        hi += 1
                    else:
                        hl -= 1

            hpropensity = float(hi + hl + f) / float(hi - hl - f)
            conn.execute("""INSERT into HFscore (adopter, value, hits, flops) VALUES ('%s', %f, %d, %d)""" %
                         (ad, hpropensity, hi, -f))
            conn.commit()

            if hpropensity < 0:
                for g in adopters_to_flop[ad]['goods']:
                    conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 0)""" % (ad, g))
            conn.commit()

            if hpropensity > 0:
                for g in adopters_to_hit[ad]:
                    conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 1)""" % (ad, g))
            conn.commit()
        conn.close()


    def __build_adopter_to_innovator(self, hits_train, goods_to_threshold):
        conn = sqlite3.connect("%s" % self.database_name)
        adopter_as_innovator = {}

        for hit in hits_train:
            cur = conn.cursor()
            cur.execute("""SELECT * from adoptions where good='%s';""" % hit)
            adoptions = cur.fetchall()
            for a in adoptions:
                if a[1] not in adopter_as_innovator:
                    adopter_as_innovator[a[1]] = {}
                    adopter_as_innovator[a[1]]["innovator"] = 0
                    adopter_as_innovator[a[1]]["notInnovator"] = 0
                    adopter_as_innovator[a[1]]["insuccesso"] = 0
                    adopter_as_innovator[a[1]]['goods'] = {}

                if hit not in adopter_as_innovator[a[1]]['goods']:
                    if int(goods_to_threshold[a[0]]) < int(a[2]):
                        adopter_as_innovator[a[1]]["notInnovator"] += 1
                    else:
                        adopter_as_innovator[a[1]]["innovator"] += 1
                        adopter_as_innovator[a[1]]['goods'][hit] = 0
        conn.close()
        return adopter_as_innovator

    def __compute_hf_propensity(self, adopters_to_hits, adopters_to_flops):
        goods_count = len(self.hits_train)
        adopters_to_epsilon_mu = {}
        for adopter in adopters_to_hits:
            adopters_to_epsilon_mu[adopter] = {}
            if adopter not in adopters_to_flops:
                adopters_to_epsilon_mu[adopter]["mu"] = 0.0
            else:
                adopters_to_epsilon_mu[adopter]["mu"] = float(adopters_to_flops[adopter]['insuccesso']) / float(goods_count)
            if adopters_to_hits[adopter]["innovator"] + adopters_to_hits[adopter]["notInnovator"] != 0:
                adopters_to_epsilon_mu[adopter]["epsilon"] = float((adopters_to_hits[adopter]["innovator"] -
                                                                 adopters_to_hits[adopter]["notInnovator"])) / \
                                                          float((adopters_to_hits[adopter]["innovator"] +
                                                                 adopters_to_hits[adopter]["notInnovator"]))
            else:
                adopters_to_epsilon_mu[adopter]["epsilon"] = 0.0

        for adopter in adopters_to_epsilon_mu:
            adopters_to_epsilon_mu[adopter]["misura"] = adopters_to_epsilon_mu[adopter]["epsilon"] * (
                1 - adopters_to_epsilon_mu[adopter]["mu"])

        return adopters_to_epsilon_mu


class WMSC(object):
    def __init__(self, database_name, innovators=True):
        self.database_name = database_name
        self.innovators = innovators

    def read_network(self):
        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        if self.innovators:
            cur.execute("""SELECT * from Coverage where hit=1;""")
        else:
            cur.execute("""SELECT * from Coverage where hit=0;""")
        edges = cur.fetchall()

        g = nx.DiGraph()
        for e in edges:
            g.add_edge(e[0], e[1])

        auth = {}
        converged = False
        limit = 100
        # Possible convergence issue
        while not converged:
            try:
                hub, auth = nx.hits(g, max_iter=limit, tol=1)
                converged = True
            except:
                limit += 50
                print "HITS Convergence issue: retry"

        adopters = sorted(auth.items(), key=operator.itemgetter(1))
        adopters.reverse()

        class_type = "hit"
        if not self.innovators:
            class_type = "flop"

        conn.execute("""CREATE TABLE IF NOT EXISTS HubsAuth_%s
                        (adopter TEXT NOT NULL
                         );""" % class_type)

        step = float(len(adopters)) / 10
        count = 0

        for covered in xrange(1, 11):
            for a in adopters:
                if count < covered * step:
                    conn.execute("""INSERT INTO HubsAuth_%s (adopter) VALUES (%s)""" % (class_type, a[0]))
                    count += 1
        conn.commit()
        conn.close()

    def __read_data(self):

        conn = sqlite3.connect("%s" % self.database_name)

        cur = conn.cursor()
        if self.innovators:
            cur.execute("""SELECT * from HFscore where value>0;""")
        else:
            cur.execute("""SELECT * from HFscore where value<0;""")
        adopters_hf = cur.fetchall()

        adopter_to_hf = {a[0]: a[1] for a in adopters_hf}

        cur = conn.cursor()
        if self.innovators:
            cur.execute("""SELECT * from Coverage where hit=1;""")
        else:
            cur.execute("""SELECT * from Coverage where hit=0;""")
        edges = cur.fetchall()

        node_covered = {}
        product_covered = {}
        for r in edges:
            product = r[1]
            adopter = r[0]

            if adopter not in adopter_to_hf:
                continue

            if product not in product_covered:
                product_covered[product] = [adopter]
            else:
                product_covered[product].append(adopter)

            if adopter not in node_covered:
                node_covered[adopter] = [[product, ], 1, float(adopter_to_hf[adopter])]
            else:
                old = node_covered[adopter]
                novel = old[0]
                novel.append(product)
                node_covered[adopter] = [novel, old[1] + 1, old[2]]

        conn.close()

        if self.innovators:
            return node_covered, sorted(node_covered,
                                        key=lambda k: (node_covered[k][1], node_covered[k][2])), product_covered
        else:
            return node_covered, sorted(node_covered,
                                        key=lambda k: (-node_covered[k][1], node_covered[k][2])), product_covered

    @staticmethod
    def __weight_coverage_test(sel, cov):
        """

        @param sel:
        @param cov:
        @return:
        """
        if len(sel) == 0:
            return False
        for i, v in sel.iteritems():
            if v < cov[i]:
                return False
        return True

    def execute(self):

        class_type = "hit"
        if not self.innovators:
            class_type = "flop"

        n, sn, products = self.__read_data()

        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""CREATE TABLE IF NOT EXISTS stats_%s
                                               (min_redundancy REAL NOT NULL,
                                               min_coverage      REAL NOT NULL,
                                               actual_coverage REAL NOT NULL,
                                               goods INTEGER NOT NULL,
                                               adopters INTEGER NOT NULL
                                               );""" % class_type)
        conn.commit()

        for min_redundancy in tqdm.tqdm(xrange(1, 11)):
            min_redundancy = float(min_redundancy) / 10
            for min_products_coverage in xrange(1, 11):
                min_products_coverage = float(min_products_coverage) / 10
                query = "CREATE TABLE IF NOT EXISTS res_%s_%s_%s (adopter TEXT NOT NULL);" \
                        % (class_type, min_redundancy, min_products_coverage)

                query = query.replace(".", "")

                conn.execute(query)
                conn.commit()

                product_coverage = {}
                for p in products:
                    product_coverage[p] = int(math.ceil(len(products[p]) * min_redundancy))

                node_selected = []
                product_selected = {}
                for i in sn:
                    if len(product_selected) < (min_products_coverage * len(products)) or \
                            not self.__weight_coverage_test(product_selected, product_coverage):

                        node_selected.append(i)
                        query = """INSERT INTO res_%s_%s_%s (adopter) VALUES ('%s')""" % \
                                (class_type, min_redundancy, min_products_coverage, i)
                        query = query.replace(".", "")
                        conn.execute(query)

                        for pp in n[i][0]:
                            if pp not in product_selected:
                                product_selected[pp] = 1
                            else:
                                product_selected[pp] += 1
                min_red = str(min_redundancy).replace(".", "")
                min_pc = str(min_products_coverage).replace(".", "")
                if len(products.keys()) > 0:
                    conn.execute("""INSERT INTO stats_%s (min_redundancy, min_coverage, actual_coverage, goods, adopters)
                              VALUES (%s, %s, %f, %d, %d)"""
                             % (class_type, min_red, min_pc, float(len(product_selected.keys())) / len(products.keys()),
                                len(product_selected.keys()), len(node_selected)))
                else:
                    conn.execute("""INSERT INTO stats_%s (min_redundancy, min_coverage, actual_coverage, goods, adopters)
                        VALUES (%s, %s, %f, %d, %d)"""
                                 % (class_type, min_red, min_pc,
                                    0,
                                    len(product_selected.keys()), len(node_selected)))
                conn.commit()
        conn.close()


class Indicators(object):

    def __init__(self, database_name, hits_train, flops_train, slots):
        self.database_name = database_name
        self.hits_train = hits_train
        self.flops_train = flops_train
        self.slots = slots

        self.parameter = ["01", "02",  "03", "04",  "05",  "06",  "07", "08",  "09",  "10"]

        hits, flops = self.__adoption_volumes()
        tot = hits + flops
        self.start_hits = int(float(hits)/tot * 10)
        self.start_flops = int(float(flops)/tot * 10)
        diff = max(self.start_hits, self.start_flops) - min(self.start_hits, self.start_flops)
        if self.start_hits < self.start_flops:
            self.start_hits = 5
            self.start_flops = 5 + diff
        elif self.start_hits > self.start_flops:
            self.start_flops = 5
            self.start_hits = 5 + diff
        else:
            self.start_hits = 5
            self.start_flops = 5

        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""CREATE TABLE IF NOT EXISTS model
                      (category TEXT NOT NULL,
                      table_name TEXT NOT NULL,
                      threshold REAL NOT NULL
                      );""")
        conn.commit()
        conn.close()

    def hitters(self):

        conn = sqlite3.connect("%s" % self.database_name)

        cur = conn.cursor()
        adoptions = cur.execute("""SELECT * from adoptions ORDER BY slot ASC;""")

        good_to_adoprs = {}
        for adoption in adoptions:
            if int(adoption[2]) <= self.slots:
                if adoption[0] in good_to_adoprs:
                    good_to_adoprs[adoption[0]][adoption[1]] = None
                else:
                    good_to_adoprs[adoption[0]] = {adoption[1]: None}

        good_to_adoptions = {}
        for g, k in good_to_adoprs.iteritems():
            good_to_adoptions[g] = len(k)

        good_to_adopters = {}
        for row in self.hits_train:
            if row not in good_to_adopters:
                good_to_adopters[row] = []
                results = conn.execute("""select * from adoptions where good='%s' and slot<=%d""" % (row, self.slots))
                for r in results:
                    good_to_adopters[row].append(r[1])

        out_file, selected_threshold = self.__alpha_beta(good_to_adopters, good_to_adoptions, hits=True)

        conn.execute("""INSERT INTO model (category, table_name, threshold) VALUES ('hit', '%s', %f);""" %
                     (out_file, selected_threshold))
        conn.commit()
        conn.close()

    def floppers(self):

        conn = sqlite3.connect("%s" % self.database_name)

        cur = conn.cursor()
        adoptions = cur.execute("""SELECT * from adoptions ORDER BY slot ASC;""")

        good_to_adoprs = {}
        for adoption in adoptions:
            if int(adoption[2]) <= self.slots:
                if adoption[0] in good_to_adoprs:
                    good_to_adoprs[adoption[0]][adoption[1]] = None
                else:
                    good_to_adoprs[adoption[0]] = {adoption[1]: None}

        good_to_adoptions = {}
        for g, k in good_to_adoprs.iteritems():
            good_to_adoptions[g] = len(k)

        good_to_adopters = {}
        for row in self.flops_train:
            if row not in good_to_adopters:
                good_to_adopters[row] = []
                results = conn.execute("""select * from adoptions where good='%s' and slot<=%d""" % (row, self.slots))
                for r in results:
                    good_to_adopters[row].append(r[1])

        out_file, selected_threshold = self.__alpha_beta(good_to_adopters, good_to_adoptions, hits=False)

        conn.execute("""INSERT INTO model (category, table_name, threshold) VALUES ('flop', '%s', %f);""" %
                     (out_file, selected_threshold))
        conn.commit()
        conn.close()

    def __alpha_beta(self, good_to_adopters, good_to_adoptions, hits=True):

        conn = sqlite3.connect("%s" % self.database_name)

        selected_threshold = 0
        best_fit = ""

        for alpha in self.parameter:
            for beta in self.parameter:
                adopters = {}

                if hits:
                    table_name = "res_hit_%s_%s" % (alpha, beta)
                else:
                    table_name = "res_flop_%s_%s" % (alpha, beta)

                table_name = table_name.replace(".", "")
                filtered_adopters = conn.execute("""SELECT * from %s;""" % table_name)

                for adopter in filtered_adopters:
                    adopters[adopter[0]] = None

                detail_table_name = "%s_goods" % table_name
                detail_table_name.replace(".", "")
                query = "CREATE TABLE IF NOT EXISTS %s (good TEXT NOT NULL, adoptions INTEGER NOT NULL DEFAULT 0);" % detail_table_name

                conn.execute(query)

                goods = {}
                for good in good_to_adopters:
                    control = False
                    for adopter in good_to_adopters[good]:
                        if adopter in adopters:
                            control = True
                            if good not in goods:
                                goods[good] = 1
                            else:
                                goods[good] += 1
                    if control:
                        if good_to_adoptions[good] == 0:
                            conn.execute("""INSERT INTO %s (good, adoptions) VALUES ('%s', 0)""" % (detail_table_name, good))
                        else:
                            conn.execute(
                                """INSERT INTO %s (good, adoptions) VALUES ('%s', %f)""" % (detail_table_name, good, float(goods[good])/good_to_adoptions[good]))
            conn.commit()

        error = sys.maxint
        for alpha in self.parameter:
            for beta in self.parameter:

                if hits:
                    detail_table_name = "res_hit_%s_%s_goods" % (alpha, beta)
                else:
                    detail_table_name = "res_flop_%s_%s_goods" % (alpha, beta)

                goods = conn.execute("""SELECT * from %s""" % detail_table_name)

                app = {}
                for good in goods:
                    app[good[0]] = float(good[1])

                if len(app) != 0:
                    arr = np.array(app.values())
                    threshold, mean_error = self.__percentile_selection(arr, 10, hits=hits)
                    print mean_error, error, detail_table_name
                    if mean_error <= error:
                        error = mean_error
                        if hits:
                            best_fit = "res_hit_%s_%s" % (alpha, beta)
                        else:
                            best_fit = "res_flop_%s_%s" % (alpha, beta)
                        best_fit = best_fit.replace(".", "")
                        selected_threshold = threshold
        conn.close()

        return best_fit, selected_threshold

    def __percentile_selection(self, vals, k, hits=True):
        lv = len(vals)
        x = int(float(lv) / k)

        thresholds = {}

        if hits:
            start = self.start_hits
        else:
            start = self.start_flops

        values = [float("{0:.2f}".format(l)) for l in vals]

        for p in xrange(start - 1 + 5, 90, start):
            z = x
            thresholds[p] = []
            while z < lv:
                if x > 0:
                    train = values[:x]
                else:
                    train = [values[0]]
                test = values[x:]
                pt = float("{0:.2f}".format(np.percentile(train, p)))
                FP = 0
                for v in test:
                    if v < pt:
                        FP += 1

                    thresholds[p].append((float(FP) / len(test)) - p)
                if x > 0:
                    z += x
                else:
                    z += 1

        for k, v in thresholds.iteritems():
            thresholds[k] = np.mean(v)

        thresholds = sorted(thresholds.items(), key=operator.itemgetter(1))
        t = float("{0:.2f}".format(np.percentile(values, thresholds[0][0])))
        return t, float("{0:.2f}".format(thresholds[0][1]))

    def __adoption_volumes(self):
        conn = sqlite3.connect("%s" % self.database_name)

        flop_adoptions_volume = 0
        hits_adoptions_volume = 0
        for g in self.hits_train:
            cur = conn.cursor()
            cur.execute("""SELECT sum(quantity) FROM adoptions WHERE good='%s' and slot <= %d""" %
                         (g, self.slots))
            res = cur.fetchone()
            if res[0] is not None:
                hits_adoptions_volume += int(res[0])

        for g in self.flops_train:
            cur = conn.cursor()
            cur.execute("""SELECT sum(quantity) FROM adoptions WHERE good='%s' and slot <= %d""" %
                         (g, self.slots))
            res = cur.fetchone()
            if res[0] is not None:
                flop_adoptions_volume += int(res[0])

        return hits_adoptions_volume, flop_adoptions_volume

if __name__ == "__main__":
    import argparse

    print "------------------------------------"
    print "          Superforecasters          "
    print "------------------------------------"
    print "Author: ", __author__
    print "Email:  ", __email__
    print "------------------------------------"

    parser = argparse.ArgumentParser()

    parser.add_argument('log', type=str, help='adoption_log file')
    parser.add_argument('gt', type=str, help='training set ground_truth file')
    parser.add_argument('db', type=str, help='desired db name', default="results.db")

    args = parser.parse_args()

    # Load Adoption log Data
    print "Loading data"
    l = LoadData(args.log, args.db)
    l.load()

    # Set Training classes
    print "Setting training classes"
    hits_train_set, flops_train_set = l.set_hit_flop(args.gt)

    # Compute Innovator thresholds (origianl method)
    print "Computing early adopters"
    e = EarlyAdoptersThreshold(args.db)
    goods_thresholds = e.execute()

    # Compute HFpropensity scores
    print "Computing-HF Propensity"
    hf = HFPropensity(args.db, hits_train_set, flops_train_set, goods_thresholds)
    hf.execute()

    # Coverage
    print "WMSC hits"
    w = WMSC(args.db, True)
    w.execute()

    print "WMSC flops"
    w = WMSC(args.db, False)
    w.execute()

    # Model construction
    print "Indicators selection"
    ids = Indicators(args.db, hits_train_set, flops_train_set, 4)
    print "Hitters"
    ids.hitters()
    print "Floppers"
    ids.floppers()
    print "Model generated"
