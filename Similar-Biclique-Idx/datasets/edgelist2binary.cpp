//
//  main.cpp
//  gen_bin_bigraph_for_MDBC_small
//
//  Created by kai on 14/9/2021.
//

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <queue>
#include <stack>
#include <set>
#include <map>

//#define NDEBUG // must precede cassert to disable assert.
#include <cassert>

using ui = unsigned int;

#define pb push_back
#define mp make_pair
class Utility {
public:
    static FILE *open_file(const char *file_name, const char *mode) {
        FILE *f = fopen(file_name, mode);
        if(f == nullptr) {
            printf("Can not open file: %s\n", file_name);
            exit(1);
        }

        return f;
    }

    static std::string integer_to_string(long long number) {
        std::vector<ui> sequence;
        if(number == 0) sequence.push_back(0);
        while(number > 0) {
            sequence.push_back(number%1000);
            number /= 1000;
        }

        char buf[5];
        std::string res;
        for(unsigned int i = sequence.size();i > 0;i --) {
            if(i == sequence.size()) sprintf(buf, "%u", sequence[i-1]);
            else sprintf(buf, ",%03u", sequence[i-1]);
            res += std::string(buf);
        }
        return res;
    }
};

using namespace std;

void gen(string original_graph)
{
    
    vector<ui> nodes;
    vector<pair<ui,ui> > edges;
//    FILE *f = Utility::open_file(("/data/MDBC/txt_graphs_and_old_index/mydblp/" + original_graph).c_str(), "r");
    FILE *f = Utility::open_file(original_graph.c_str(), "r");
    
    char buf[1024];
    ui t_n1, t_n2, t_m;
    
    if(fgets(buf, 1024, f) != NULL) {
        for(ui j = 0;buf[j] != '\0';j ++) if(buf[j] < '0'||buf[j] > '9') buf[j] = ' ';
        sscanf(buf, "%u%u%u", &t_n1, &t_n2, &t_m);
    }
    cout<<"1st in original graph : t_n1 = "<<t_n1<<", t_n2 = "<<t_n2<<", t_m = "<<t_m<<endl;
    
    ui a, b;
    
    while(fgets(buf, 1024, f)) {
        char comment = 1;
        for(ui j = 0;buf[j] != '\0';j ++) if(buf[j] != ' '&&buf[j] != '\t') {
            if(buf[j] >= '0'&&buf[j] <= '9') comment = 0;
            break;
        }
        if(comment) continue;

        for(ui j = 0;buf[j] != '\0';j ++) if(buf[j] < '0'||buf[j] > '9') buf[j] = ' ';
        sscanf(buf, "%u%u", &a, &b);
        if(a == b) continue;
        nodes.push_back(a);
        nodes.push_back(b);
        edges.push_back(make_pair(a,b));
        edges.push_back(make_pair(b,a));
    }

    fclose(f);
    
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

    printf("min id = %u, max id = %u, n = %lu\n", nodes.front(), nodes.back(), nodes.size());
    
    if(nodes.size() != (t_n1+t_n2)) {
        cout<<"nodes.size() != (t_n1+t_n2)"<<endl;
        exit(1);
    }
    
    sort(edges.begin(), edges.end());
    edges.erase(unique(edges.begin(), edges.end()), edges.end());

    map<ui,ui> M;
    for(ui i = 0;i < nodes.size();i ++) M[nodes[i]] = i;

    char preserved = 1;
    for(ui i = 0;i < nodes.size();i ++) if(nodes[i] != i) preserved = 0;
    if(!preserved) printf("Node ids are not preserved!\n");

    ui n = nodes.size();
    ui m = edges.size();
    printf("n = %s, m = %s\n", Utility::integer_to_string(n).c_str(), Utility::integer_to_string(m/2).c_str());
    
    if(m/2 != t_m) {
        cout<<"m/2 != t_m"<<endl;
        exit(1);
    }
    
    ui *pstart = new ui[n+1];
    ui *edge = new ui[m];

    ui j = 0;
    for(ui i = 0;i < n;i ++) {
        pstart[i] = j;
        while(j < m&&edges[j].first == nodes[i]) {
            edge[j] = M[edges[j].second];
            ++ j;
        }
    }
    pstart[n] = j;

    original_graph.erase(original_graph.end() - 4, original_graph.end());

//    f = Utility::open_file(("/data/MDBC/" + original_graph + "_b_degree.bin").c_str(), "wb");
    f = Utility::open_file((original_graph + "_b_degree.bin").c_str(), "wb");

    ui tt = sizeof(ui);
    fwrite(&tt, sizeof(ui), 1, f);
    fwrite(&t_n1, sizeof(ui), 1, f);
    fwrite(&t_n2, sizeof(ui), 1, f);
    fwrite(&m, sizeof(ui), 1, f);

    assert(n == (t_n1+t_n2));
    
    ui *degree = new ui[n];
    for(ui i = 0;i < n;i ++) degree[i] = pstart[i+1]-pstart[i];
    fwrite(degree, sizeof(ui), n, f);
    fclose(f);

//    f = Utility::open_file(("/data/MDBC/" + original_graph + "_b_adj.bin").c_str(), "wb");
    f = Utility::open_file((original_graph + "_b_adj.bin").c_str(), "wb");
                            
    fwrite(edge, sizeof(ui), m, f);
    fclose(f);
    
    //测试另外一种存图的方式，space比较大.(大概是2倍)
    /*
    f = Utility::open_file((original_graph + "_b_g.bin").c_str(), "wb");
    for(ui i = 0; i < n; i++) {
        for(ui j = pstart[i]; j < pstart[i+1]; j++){
            ui nei = edge[j];
            fwrite(&i, sizeof(ui), 1, f);
            fwrite(&nei, sizeof(ui), 1, f);
        }
    }
    fclose(f);
    */
    
    delete[] pstart;
    delete[] edge;
    delete[] degree;
    
    cout<<"finish generating binary version!"<<endl;
}

int main(int argc, const char * argv[]) {
    if(argc < 2) {
        printf("Usage: [1].exe [2].original_graph \n");
        return 0;
    }
    
    cout<<"graph name = "<<argv[1]<<endl;
    
    gen(string(argv[1]));
        
    return 0;
}
