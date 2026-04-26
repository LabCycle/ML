//11 SIMPLE MULTITHREADING: SUM + SEARCH

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace chrono;

// -------- Worker Functions --------
void partialSum(const vector<int>& arr, int s, int e, long long& res) {
    res = 0;
    for (int i = s; i < e; i++) res += arr[i];
}

void partialSearch(const vector<int>& arr, int s, int e, int key, bool& found) {
    for (int i = s; i < e; i++)
        if (arr[i] == key) { found = true; return; }
}

int main() {

    int n, key;
    cout << "Enter size & key: ";
    cin >> n >> key;

    // Generate data
    vector<int> arr(n);
    for (int &x : arr) x = rand() % 100000;

    // -------- Sequential --------
    auto t1 = high_resolution_clock::now();

    long long seq_sum = 0;
    bool seq_found = false;

    for (int x : arr) {
        seq_sum += x;
        if (x == key) seq_found = true;
    }

    auto t2 = high_resolution_clock::now();

    // -------- Parallel --------
    int threadsCount = 4;
    vector<thread> threads;
    vector<long long> sums(threadsCount);
    vector<bool> found(threadsCount, false);

    int chunk = n / threadsCount;

    auto t3 = high_resolution_clock::now();

    // SUM
    for (int i = 0; i < threadsCount; i++) {
        int s = i * chunk;
        int e = (i == threadsCount - 1) ? n : s + chunk;
        threads.emplace_back(partialSum, cref(arr), s, e, ref(sums[i]));
    }

    for (auto &t : threads) t.join();

    long long par_sum = 0;
    for (auto s : sums) par_sum += s;

    threads.clear();

    // SEARCH
    for (int i = 0; i < threadsCount; i++) {
        int s = i * chunk;
        int e = (i == threadsCount - 1) ? n : s + chunk;
        threads.emplace_back(partialSearch, cref(arr), s, e, key, ref(found[i]));
    }

    for (auto &t : threads) t.join();

    bool par_found = false;
    for (auto f : found) if (f) par_found = true;

    auto t4 = high_resolution_clock::now();

    // -------- Output --------
    auto seq_time = duration_cast<milliseconds>(t2 - t1).count();
    auto par_time = duration_cast<milliseconds>(t4 - t3).count();

    cout << "\nSequential Sum: " << seq_sum
         << "\nParallel Sum: " << par_sum;

    cout << "\nKey Found (Seq): " << (seq_found ? "Yes" : "No")
         << "\nKey Found (Par): " << (par_found ? "Yes" : "No");

    cout << "\nSequential Time: " << seq_time << " ms";
    cout << "\nParallel Time: " << par_time << " ms";

    double speedup = (double)seq_time / par_time;
    cout << "\nSpeedup: " << speedup;
    cout << "\nEfficiency: " << speedup / threadsCount << endl;

    return 0;
}
