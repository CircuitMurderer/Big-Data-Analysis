import os
import pandas as pd
from senti import Senti
from collections import Counter


def favor_comment_cur(count: Counter):
    score = count[1] * 1 + count[0] * .5 + count[-1] * 0
    rate = score / sum(count.values())
    return rate


def favor_comment_all(s: str):
    return int(s.split(';')[-1].split('%')[0]) * 0.01


def leave_float(n: float):
    return float('{:.2f}'.format(n))


def parse_name(prod_name: str):
    this_knd = ''
    p_name = prod_name.split('_')
    others = ['4G', '5G', '(', 'GB', '骁龙', '天玑', '设计', '清', '充', '摄', '电']
    kinds = ['Redmi', 'vivo', '荣耀', 'Apple', 'HUAWEI', 'OPPO', '小米', '三星', '索尼']

    for nm in p_name:
        for kind in kinds:
            if kind in nm:
                this_knd = kind
                break
        if this_knd != '':
            break

    for i, nm in enumerate(p_name):
        for other in others:
            if other in nm:
                return ' '.join(p_name[:i]).strip(' '), this_knd

    assert this_knd != '', 'Cannot recognize the kind of this product.'
    return ' '.join(p_name).strip(' '), this_knd


if __name__ == '__main__':
    sen = Senti()
    res_path = './result/'
    comm_path = './comments/'
    columns = ['型号', '品牌', '最近好评率', '历史好评率', '合计好评率']

    if not os.listdir(comm_path):
        raise FileNotFoundError("No comments file found. Please run 'crawl.py' first.")

    df = []
    dirs = os.walk(comm_path)
    for dr, _, fl in dirs:
        if not fl:
            continue
        # print(dr)
        now_df = []
        for f in fl:
            pr_name, pr_kind = parse_name(f)
            pth = '/'.join([dr, f])
            pred = sen.predict(pth)
            cnt = Counter(pred)
            his_fav = favor_comment_all(f)
            cur_fav = favor_comment_cur(cnt)
            all_fav = .7 * cur_fav + .3 * his_fav
            # print("{:.2f}, {:.2f}, {:.2f}, {}".format(rec_fav, his_fav, .7 * rec_fav + .3 * his_fav, name))

            now_df.append((pr_name, pr_kind, leave_float(cur_fav),
                           leave_float(his_fav), leave_float(all_fav)))

        df.extend(now_df)
        now_df = pd.DataFrame(now_df, columns=columns)
        now_df.to_csv(res_path + dr.strip('./').split('/')[-1] + '.csv', index=False)

    df = pd.DataFrame(df, columns=columns)
    # df = df.sort_values(by='合计好评率', ascending=False)
    df.to_csv(res_path + 'all.csv', index=False)
