#include <bits/stdc++.h>

using namespace std;

struct date {
    int year, month, day;

    date(): year(0), month(0), day(0) {}

    date(string &s) {
        year = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0'),
        month = (s[5] - '0') * 10 + (s[6] - '0'),
        day = (s[8] - '0') * 10 + (s[9] - '0');
    }

    bool operator <(const date &other) const {
        if (year != other.year) {
            return year < other.year;
        } else if (month != other.month) {
            return month < other.month;
        } else {
            return day < other.day;
        }
    }
};

struct raw {
    float sum_b, percent;
    date d;
    string other, id;

    raw (vector<string> &s) {
        d = date(s[2]);
        sum_b = s[6] == "" ? 0: stof(s[6]);
        percent = s[14] == "" ? 0: stof(s[14]);
        id = s[8];
        other = s[1] + ',' + s[3] + ',' + s[4] + ',' + s[5] + ',' +
                s[7] + ',' + s[9] + ',' + s[10] + ',' + s[11] + ',' +
                s[12] + ',' + s[13] + ',' + s[15];
    }

    bool operator <(const raw &other) {
        return d < other.d;
    }
};

void split(vector<string> *splitted, string &s) {
    string cur = "";
    for (char c: s) {
        if (c == ',') {
            splitted->push_back(cur);
            cur = "";
        } else {
            cur += c;
        }
    }
    splitted->push_back(cur);
}

void print_columns(vector<string> &s) {
    cout << s[1] << ',' << s[3] << ',' << s[4] << ',' << s[5] <<
            ',' << s[7] << ',' << s[9] << ',' << 
            s[10] << ',' << s[11] << ',' << s[12] << ',' << s[13] <<
            ',' << s[15] << ',';
    cout << s[2] << ',';
    cout << s[8] << ','  << s[6] << ',' << s[14] << ',';
    cout << "cur_points" << '\n';
}

void print_transaction(raw & trans, float cur_points) {
    cout << trans.other << ',';
    cout << trans.d.year << '-' << trans.d.month << '-' << trans.d.day << ',';
    cout << trans.id << ',' << trans.sum_b << ',' << trans.percent << ',';
    cout << cur_points << '\n';
}

float get_bonus(float last_month_spend, float sum_b) {
    if (last_month_spend >= 10000) {
        return 0.05 * sum_b;
    } else if (last_month_spend >= 8000) {
        return 0.04 * sum_b;
    } else if (last_month_spend >= 6000) {
        return 0.03 * sum_b;
    } else {
        return 0;
    }
}

vector<raw> transaction;
unordered_map<string, float> points,
                            spend[2];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);
    auto fread = freopen("./data/train_data.csv", "r", stdin);
    auto fwrite = freopen("./data/train_points.csv", "w", stdout);
    auto start = clock();
    vector<string> splitted;
    string columns;
    getline(cin, columns);
    split(&splitted, columns);
    print_columns(splitted); 
    string s;
    while(getline(cin, s)) {
        splitted.clear();
        split(&splitted, s);
        transaction.push_back(raw(splitted));
    }
    sort(transaction.begin(), transaction.end());
    int cur_month = transaction[0].d.month,
        cur_map = 0;
    for (auto trans: transaction) {
        if (trans.d.month != cur_month) {
            cur_month = trans.d.month;
            cur_map = 1 - cur_map;
            spend[cur_map].clear();
        }
        spend[cur_map][trans.id] += trans.sum_b;
        points[trans.id] -= trans.percent;
        points[trans.id] += get_bonus(spend[1 - cur_map][trans.id], trans.sum_b);
        print_transaction(trans, points[trans.id]);
    }
    cout << endl;
    freopen ("/dev/tty", "a", stdout);
    // cout << transaction[0].d.year << '-' << transaction[0].d.month <<
    //         '-' << transaction[0].d.day << endl;
    // cout << transaction.back().d.year << '-' << transaction.back().d.month <<
    //         '-' << transaction.back().d.day << endl;
    cout << 1. * (clock() - start) / CLOCKS_PER_SEC << endl;
    return 0;
}