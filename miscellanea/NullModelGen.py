import random

__author__ = 'GiulioRossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


class NullModel(object):

    def __init__(self, filename, out_filename):
        self.filename = filename
        self.out_filename = out_filename

    def execute(self):

        f = open("%s" % self.filename)

        adopter_to_slot = {}
        adopters = {}

        for row in f:

            row = row.rstrip().split(",")

            k = "%s,%s" % (row[0], row[2])

            if k not in adopter_to_slot:
                adopter_to_slot[k] = int(row[3])
            else:
                adopter_to_slot[k] += int(row[3])

            ad = int(row[1])
            if ad not in adopters:
                adopters[ad] = int(row[3])
            else:
                adopters[ad] += int(row[3])

        al_ad_slot = []
        al_us = []

        for u in adopter_to_slot:
            al_ad_slot.append([u, adopter_to_slot[u]])

        for u in adopters:
            al_us.append([u, adopters[u]])

        finale = {}

        while len(al_ad_slot) != 0:
            now_us = random.randint(0, len(al_us)-1)
            cur_us = al_us[now_us][0]
            al_us[now_us][1] -= 1
            if al_us[now_us][1] == 0:
                b = al_us[:now_us]
                b.extend(al_us[now_us+1:])
                al_us = b

            now_ad_slot = random.randint(0, len(al_ad_slot)-1)
            cur_ar_mo = al_ad_slot[now_ad_slot][0]
            al_ad_slot[now_ad_slot][1] -= 1
            if al_ad_slot[now_ad_slot][1] == 0:
                a = al_ad_slot[:now_ad_slot]
                a.extend(al_ad_slot[now_ad_slot+1:])
                al_ad_slot = a

            k = "%s,%s,%s" % (cur_ar_mo.split(",")[0], cur_us, cur_ar_mo.split(",")[1])
            if k not in finale:
                finale[k] = 1
            else:
                finale[k] += 1

        out = open("%s" % self.out_filename, "w")

        c = 0

        for k, v in finale.iteritems():
            c += 1
            out.write("%s,%s\n" % (k, v))
            if c % 1000 == 0:
                out.flush()

        out.flush()
        out.close()


if __name__ == "__main__":
    import argparse

    print "------------------------------------"
    print "             Null Model             "
    print "------------------------------------"
    print "Author: ", __author__
    print "Email:  ", __email__
    print "------------------------------------"

    parser = argparse.ArgumentParser()

    parser.add_argument('log', type=str, help='adoption_log')
    parser.add_argument('out', type=str, help='out file name')

    args = parser.parse_args()
    g = NullModel(args.log, args.out)
    g.execute()
