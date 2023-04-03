from SNA.depression.measures import print_functions as pf

DEPR_SCORE = 'depression.L1'
DEPR_ZONE = 'depression.zone'
TIME = 'n.sec.h'
DYADIC = 'n.sec.dyad.h'
RATIO = 'ratio'

if __name__ == '__main__':
    pf['correlation coefficient between attributes'](DEPR_SCORE, TIME)
    pf['correlation coefficient between attributes'](DEPR_SCORE, RATIO)
    pf['correlation coefficient between attribute and degree centrality'](DEPR_SCORE)
    pf['homophily for a metric'](DEPR_SCORE)
    pf['homophily for a metric'](DEPR_ZONE)
    pf['mean time spent and zones']()
