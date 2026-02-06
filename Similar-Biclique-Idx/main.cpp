//
//  main.cpp
//  MDBC
//
//  Created by kai on 20/4/2021.
//

#include "Timer.h"
#include "Utility.h"
#include "LinearHeap.h"


#define MAT_WAY2

int maxTime = 3600;  //seconds
int maxdurtime = 10;
double each_start_time;
Timer durt;
bool over_time_flag = false;
double startTime;
double durTime;
int binary_version;
int keyid;
long long total_num_idx = 0;
long long unskipped_num_idx = 0;

double STR_start_time;
int STR_max_time = 60;

ui n, n1, n2, m;
ui * pstart;
ui * edges;
int * degree; //vertex degree, const!

int * TMPdeg; //used in vertex reduction
int * Sdeg; //used in vertex reduction

int * comneicntforDBLP;

int * os;

vector<vector<ui>> G_vv;
unordered_map<ui, unordered_set<ui>> G_map;
vector<pair<ui, ui>> new_edges_vec;

//the remaining graph
ui r_n, r_n1, r_n2, r_m;
ui * r_pstart;
ui * r_edges;
int * r_degree;
ui * peel_s;
ui * oid;
ui * nid;

//vector<vector<ui>> VLvnei;

map<ui, vector<ui>> VLvnei;

ui max_CS_deg;  //used to construct Matrix
int ** Matrix;
ui * trans;
int * inPQ;
int * deg_inP;
int * pdeg_inP;
int * ndeg_inP;

pair<ui, ui> * LRnid;

bool * del_ver;
bool * del_ver_copy;
ui * CLp;  //Candidate List position
bool * CLe;

int * inCR;
int * domCR;
int * deg_inCR;
int skipped_dom_ver;
int total_ver;

dense_hash_map<ui, int> * sup; //butterfly count
dense_hash_map<ui, bool> * edel; //0:exist, 1:deleted
dense_hash_map<ui, bool> * esign; //exist

int vr_way;
int update_type;
int num_of_update;
//int enum_way;
double epsilon;
int tau;
int noRSim;
int Rtau = 2;

int domway = 2;

int thre_make_seg = 12;
double trivial_score = 0;
double seg_num_times;
double rg_limit;

vector<pair<vector<ui>, vector<ui>>> results;
long long MDBC_num = 0;
int max_MDBC_size = 0;
int min_MDBC_size = INF;

long long cnt_prune_by_ub = 0;
long long cnt_cmp_by_js = 0;
long long total_js = 0;

double min_density;
double max_density;
double ave_density;
long long ave_density_cnt;

long long BTNK_SimNei_T = 0;

long long preEnum_initSet = 0;
long long preEnum_buildMatrx = 0;
long long preEnum_Enum = 0;

long long enumT_Maximal_ET = 0;
long long enumT_Dom = 0;
long long enumT_Sort = 0;
long long enumT_Recur = 0;

unordered_map<ui, unordered_map<ui, int>> hashG;

bool cmp_degP (ui u, ui v)
{
    return deg_inP[u] < deg_inP[v];
}

bool cmp_degCR (ui u, ui v)
{
    return deg_inCR[u] < deg_inCR[v];
}

bool cmp_Sdeg (ui u, ui v)
{
    return Sdeg[oid[u]] < Sdeg[oid[v]];
}

class Itval
{
public:
        ui s_idx; //true vertex id
        ui e_idx; //true vertex id
        double min_score;
        double max_score;
        int c; //这个range中真实有效的个数
    Itval() {
        s_idx = 0;
        e_idx = 0;
        min_score = 0;
        max_score = 0;
        c = 0;
    }
    Itval(ui _s, ui _e, double _mins, double _maxs, int _c){
        s_idx = _s;
        e_idx = _e;
        min_score = _mins;
        max_score = _maxs;
        c = _c;
    }
};

class Range
{
public:
    ui coreV; //基点的vertex id
    int rgC; //L vertex id ~ R vertex id
    ui Lidx; //在ordered 2hop neighbors这个list中的 L position
    ui Ridx; //在ordered 2hop neighbors这个list中的 R position
    Range() {
        coreV = 0;
        rgC = 0;
        Lidx = 0;
        Ridx = 0;
    }
    Range(ui _coreV, int _rgC, ui _Lidx, ui _Ridx){
        coreV = _coreV;
        rgC = _rgC;
        Lidx = _Lidx;
        Ridx = _Ridx;
    }
};

class node
{
public:
    node * L;
    node * R;
    node * P;
    double mins;
    double maxs;
    int mark;
    int isv;
    int idx;
    node() {
        L = nullptr;
        R = nullptr;
        P = nullptr;
        mins = 0.0;
        maxs = 0.0;
        mark = 0;
        isv = 0;
        idx = 0;
    }
    node(node * _L, node * _R, node * _P, double _mins, double _maxs, int _mark, int _isv, int _idx) {
        L = _L;
        R = _R;
        P = _P;
        mins = _mins;
        maxs = _maxs;
        mark = _mark;
        isv = _isv;
        idx = _idx;
    }
};

vector<vector<Itval>> vsn;
vector<vector<pair<ui, double>>> index_storeall;

void load_graph_binary(string graph_name)
{
    
    cout<<"start reading graph "<<graph_name<<endl;
    graph_name.erase(graph_name.end()-4, graph_name.end());
    FILE *f = Utility::open_file((graph_name + string("_b_degree.bin")).c_str(), "rb");

    ui tt;
    fread(&tt, sizeof(ui), 1, f);
    if(tt != sizeof(ui)) {
        printf("sizeof unsigned int is different: b_degree.bin(%u), machine(%lu)\n", tt, sizeof(ui));
        return ;
    }
    
    fread(&n1, sizeof(ui), 1, f);
    fread(&n2, sizeof(ui), 1, f);
    n = n1 + n2;
    fread(&m, sizeof(ui), 1, f);

    printf("*\t 1st line: n1=%s, n2=%s, n=%s, m (2x edges)=%s \n", Utility::integer_to_string(n1).c_str(), Utility::integer_to_string(n2).c_str(), Utility::integer_to_string(n).c_str(), Utility::integer_to_string(m).c_str());

    ui *deg = new ui[n];
    fread(deg, sizeof(ui), n, f);
    
    fclose(f);

#ifndef NDEBUG
    long long sum = 0;
    for(ui i = 0;i < n;i ++) sum += deg[i];
    assert(sum == m);
#endif

    f = Utility::open_file((graph_name + string("_b_adj.bin")).c_str(), "rb");

    pstart = new ui[n+1];
    edges = new ui[m];
    degree = new int[n];
    TMPdeg = new int[n];
    Sdeg = new int[n];
    os = new int[n];
    comneicntforDBLP = new int[n];
    memset(comneicntforDBLP, 0, sizeof(int)*n);

    pstart[0] = 0;
    for(ui i = 0;i < n;i ++) {
        if(deg[i] > 0) fread(edges+pstart[i], sizeof(ui), deg[i], f);
        pstart[i+1] = pstart[i] + deg[i];
    }

    fclose(f);

    for(ui i = 0; i < n; i++) {
        degree[i] = deg[i];
        TMPdeg[i] = deg[i];
        Sdeg[i] = 0;
    }
    
    for(ui i = 0; i < n1; i++) os[i] = 1;
    for(ui i = n1; i < n; i++) os[i] = 2;
    
    delete[] deg;
    
#ifdef _CaseStudy_
    for(ui u = 0; u < n; u++) {
        for(ui j = pstart[u]; j < pstart[u+1]; j++) {
            ui v = edges[j];
            hashG[u][v] = 1;
            hashG[v][u] = 1;
        }
    }
#endif

}

void load_graph(string graph_name)
{
    ifstream input_file(graph_name, ios::in);
    map<ui, set<ui>> biG;
    if (!input_file.is_open()){
        cout << "cannot open file : "<<graph_name<<endl;exit(1);
    }
    else{
        input_file >> n1 >> n2 >> m;
        n = n1 + n2;
        cout<<graph_name<<" : n1 = "<<n1<<", n2 = "<<n2<<", m = "<<m<<endl;
        ui tu, tv;
        while (input_file >> tu >> tv) {
            assert(tu != tv);
            assert(tu >= 0 && tu < n);
            assert(tv >= 0 && tv < n);
            biG[tu].insert(tv);
            biG[tv].insert(tu);
        }
        assert(biG.size() == n);
        m = 0;
        for(auto e : biG) m += e.second.size();
        assert(m%2 == 0); m /= 2;
        cout<<"after loading, n = "<<n<<", m = "<<m<<", ";
        input_file.close();
    }
    
    pstart = new ui[n+1];
    edges = new ui[2*m];
    degree = new int[n];
    
    TMPdeg = new int[n];
    Sdeg = new int[n];
    
    CLp = new ui[n];
    memset(CLp, 0, sizeof(ui)*n);
    CLe = new bool[n];
    memset(CLe, 0, sizeof(bool)*n);
        
    pstart[0] = 0;
    for(ui i = 0; i < n; i++){
        const set<ui> & nei = biG[i];
        ui s_idx = pstart[i];
        for(auto e : nei) edges[s_idx++] = e;
        pstart[i+1] = s_idx;
        degree[i] = nei.size();
        TMPdeg[i] = nei.size();
        Sdeg[i] = 0;
    }
    assert(pstart[n] == 2*m);
}

void load_index_LG(string graph_name)
{
//    cout<<" *** in load_index_LG ***"<<endl;
    if(rg_limit != 0) {cout<<"in LG, rg_limit must be 0!"<<endl;exit(1);}
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_LG.bin");

    cout<<"index name : "<<graph_name<<endl;
    
    FILE * f = Utility::open_file(graph_name.c_str(), "rb");
    vsn.resize(n);
    
    int vid, numofI;
    ui s_vid, e_vid;
    char minS, maxS;
    int cnt;
    unsigned long result;
    
    while(fread(&vid, sizeof(int), 1, f) == 1) {
//        cout<<"vid = "<<vid<<endl;
        fread(&numofI, sizeof(int), 1, f);
//        cout<<"numofI = "<<numofI<<endl;
//        cout.flush();
        vector<Itval> & tmp_vec = vsn[vid];
        if(numofI > 0){
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&e_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                fread(&maxS, sizeof(char), 1, f);
                fread(&cnt, sizeof(int), 1, f);
                tmp_vec.push_back(Itval(s_vid, e_vid, (double)minS/100, (double)maxS/100, cnt));
//                cout<<"   +  "<<s_vid<<","<<e_vid<<","<<(double)minS/100<<","<<(double)maxS/100<<","<<cnt<<endl;
                -- numofI;
                if(numofI == 0) break;
            }
        }
        else { //indi
            numofI = -numofI;
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                
                tmp_vec.push_back(Itval(s_vid, s_vid, (double)minS/100, (double)minS/100, 1));
//                cout<<"   -  "<<s_vid<<","<<(double)minS/100<<","<<endl;
                
                -- numofI;
                if(numofI == 0) break;
            }
        }
    }
    fclose(f);
//    cout<<"*** finish load_index_LG.bin ***"<<endl;
        

    /*************************txt */
//    graph_name.erase(graph_name.end() - 4, graph_name.end());
//    graph_name.append("_" + to_string((int)(trivial_score*100)));
//    graph_name.append("_" + to_string((int)(thre_make_seg)));
//    graph_name.append("_" + to_string((int)(seg_num_times*100)));
//    graph_name.append("_" + to_string((int)(rg_limit*100)));
//    graph_name.append("_LG.txt");
//
//    cout<<"index name : "<<graph_name<<endl;
//
//    ifstream fin;
//    fin.open(graph_name);
//    if(!fin.is_open()){
//        cout<<"cannot open index LG."<<endl; exit(1);
//    }
//
//    vsn.resize(n);
//
//    string line;
//    int vid;
//    ui s_vid, e_vid, cnt;
//    double minS, maxS;
//    while (getline(fin, line)) {
//        stringstream ss(line);
//        ss >> vid;
//        if(vid >= 0) {
//            vector<Itval> & tmp_vec = vsn[vid];
//            while (ss >> s_vid >> e_vid >> minS >> maxS >> cnt) {
//                tmp_vec.push_back(Itval(s_vid, e_vid, minS/100, maxS/100, cnt));
//            }
//        }
//        else {
//            vid = -vid;
//            vector<Itval> & tmp_vec = vsn[vid];
//            while (ss >> s_vid >> minS) {
//                tmp_vec.push_back(Itval(s_vid, s_vid, minS/100, minS/100, 1));
//            }
//        }
//    }
//    fin.close();
//    cout<<"*** finish load_index_LG.txt ***"<<endl;
    
    /*************************txt */

//        for(ui i = 0; i < n; i++){
//            cout<<"for vertex "<<i<<endl;
//            for(auto x : vsn[i]) {
//                cout<<"     "<<x.s_idx<<","<<x.e_idx<<","<<x.min_score<<","<<x.max_score<<","<<x.c<<endl;
//            }
//        }

}

void load_index_LGf(string graph_name)
{
//    cout<<" *** in load_index_LGf ***"<<endl;
    if(rg_limit != 0) {cout<<"in LGf, rg_limit must be 0!"<<endl;exit(1);}
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_LGf.bin");

    cout<<"index name : "<<graph_name<<endl;
    
    FILE * f = Utility::open_file(graph_name.c_str(), "rb");
    vsn.resize(n);
    
    int vid, numofI;
    ui s_vid, e_vid;
    char minS, maxS;
    int cnt;
    unsigned long result;
    
    while(fread(&vid, sizeof(int), 1, f) == 1) {
//        cout<<"vid = "<<vid<<endl;
        fread(&numofI, sizeof(int), 1, f);
//        cout<<"numofI = "<<numofI<<endl;
//        cout.flush();
        vector<Itval> & tmp_vec = vsn[vid];
        if(numofI > 0){
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&e_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                fread(&maxS, sizeof(char), 1, f);
                fread(&cnt, sizeof(int), 1, f);
                tmp_vec.push_back(Itval(s_vid, e_vid, (double)minS/100, (double)maxS/100, cnt));
//                cout<<"   +  "<<s_vid<<","<<e_vid<<","<<(double)minS/100<<","<<(double)maxS/100<<","<<cnt<<endl;
                -- numofI;
                if(numofI == 0) break;
            }
        }
        else { //indi
            numofI = -numofI;
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                
                tmp_vec.push_back(Itval(s_vid, s_vid, (double)minS/100, (double)minS/100, 1));
//                cout<<"   -  "<<s_vid<<","<<(double)minS/100<<","<<endl;
                
                -- numofI;
                if(numofI == 0) break;
            }
        }
    }
    fclose(f);
//    cout<<"*** finish load_index_LGf.bin ***"<<endl;
        

    /*************************txt */
//    graph_name.erase(graph_name.end() - 4, graph_name.end());
//    graph_name.append("_" + to_string((int)(trivial_score*100)));
//    graph_name.append("_" + to_string((int)(thre_make_seg)));
//    graph_name.append("_" + to_string((int)(seg_num_times*100)));
//    graph_name.append("_" + to_string((int)(rg_limit*100)));
//    graph_name.append("_LG.txt");
//
//    cout<<"index name : "<<graph_name<<endl;
//
//    ifstream fin;
//    fin.open(graph_name);
//    if(!fin.is_open()){
//        cout<<"cannot open index LG."<<endl; exit(1);
//    }
//
//    vsn.resize(n);
//
//    string line;
//    int vid;
//    ui s_vid, e_vid, cnt;
//    double minS, maxS;
//    while (getline(fin, line)) {
//        stringstream ss(line);
//        ss >> vid;
//        if(vid >= 0) {
//            vector<Itval> & tmp_vec = vsn[vid];
//            while (ss >> s_vid >> e_vid >> minS >> maxS >> cnt) {
//                tmp_vec.push_back(Itval(s_vid, e_vid, minS/100, maxS/100, cnt));
//            }
//        }
//        else {
//            vid = -vid;
//            vector<Itval> & tmp_vec = vsn[vid];
//            while (ss >> s_vid >> minS) {
//                tmp_vec.push_back(Itval(s_vid, s_vid, minS/100, minS/100, 1));
//            }
//        }
//    }
//    fin.close();
//    cout<<"*** finish load_index_LG.txt ***"<<endl;
    
    /*************************txt */

//        for(ui i = 0; i < n; i++){
//            cout<<"for vertex "<<i<<endl;
//            for(auto x : vsn[i]) {
//                cout<<"     "<<x.s_idx<<","<<x.e_idx<<","<<x.min_score<<","<<x.max_score<<","<<x.c<<endl;
//            }
//        }

}

void load_index_GRL(string graph_name)
{
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL.txt");
    ifstream fin;
    fin.open(graph_name);
    if(!fin.is_open()){
        cout<<"cannot open index GRL."<<endl; exit(1);
    }
    
    vsn.resize(n);
    
    string line;
    ui vid, s_vid, e_vid, cnt;
    double minS, maxS;
    while (getline(fin, line)) {
        stringstream ss(line);
        ss >> vid;
        assert(vid >=0 && vid < n);
        vector<Itval> & tmp_vec = vsn[vid];
//        cout<<" a new line, "<<vid<<": ";
        while (ss >> s_vid >> e_vid >> minS >> maxS >> cnt) {
//            cout<<s_vid<<", "<<e_vid<<", "<<minS<<", "<<maxS<<", "<<cnt<<" ";
            tmp_vec.push_back(Itval(s_vid, e_vid, minS, maxS, cnt));
        }
//        cout<<endl;
    }
    fin.close();
    cout<<"*** finish load_index_GRL ***"<<endl;
//    cout<<"vsn : "<<endl;
//    for(ui i = 0; i < n; i++){
//        vector<Itval> & tmp_vec = vsn[i];
//        cout<<"v "<<i<<" : ";
//        for(auto e : tmp_vec){
//            cout<<e.s_idx<<","<<e.e_idx<<","<<e.min_score<<","<<e.max_score<<","<<e.c<<"  ";
//        }
//        cout<<endl;
//    }
}

void load_index_GRL2(string graph_name)
{
//    cout<<" *** in load_index_GRL2 ***"<<endl;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_GRL2.bin");

    cout<<"index name : "<<graph_name<<endl;
    
    FILE * f = Utility::open_file(graph_name.c_str(), "rb");
    vsn.resize(n);
    
    int vid, numofI;
    ui s_vid, e_vid;
    char minS, maxS;
    int cnt;
    unsigned long result;
    
    while(fread(&vid, sizeof(int), 1, f) == 1) {
//        cout<<"vid = "<<vid<<endl;
        fread(&numofI, sizeof(int), 1, f);
//        cout<<"numofI = "<<numofI<<endl;
//        cout.flush();
        vector<Itval> & tmp_vec = vsn[vid];
        if(numofI > 0){
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&e_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                fread(&maxS, sizeof(char), 1, f);
                fread(&cnt, sizeof(int), 1, f);
                tmp_vec.push_back(Itval(s_vid, e_vid, (double)minS/100, (double)maxS/100, cnt));
//                cout<<"   +  "<<s_vid<<","<<e_vid<<","<<(double)minS/100<<","<<(double)maxS/100<<","<<cnt<<endl;
                -- numofI;
                if(numofI == 0) break;
            }
        }
        else { //indi
            numofI = -numofI;
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                
                tmp_vec.push_back(Itval(s_vid, s_vid, (double)minS/100, (double)minS/100, 1));
//                cout<<"   -  "<<s_vid<<","<<(double)minS/100<<","<<endl;
                
                -- numofI;
                if(numofI == 0) break;
            }
        }
    }
    fclose(f);
//    cout<<"*** finish load_index_GRL2.bin ***"<<endl;
    
    //txt version
    /*
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL2.txt");
    ifstream fin;
    fin.open(graph_name);
    if(!fin.is_open()){
        cout<<"cannot open index GRL2."<<endl; exit(1);
    }
    
    vsn.resize(n);
    
    string line;
    ui vid, s_vid, e_vid, cnt;
    double minS, maxS;
    while (getline(fin, line)) {
        stringstream ss(line);
        ss >> vid;
        assert(vid >=0 && vid < n);
        vector<Itval> & tmp_vec = vsn[vid];
//        cout<<" a new line, "<<vid<<": ";
        while (ss >> s_vid >> e_vid >> minS >> maxS >> cnt) {
//            cout<<s_vid<<", "<<e_vid<<", "<<minS<<", "<<maxS<<", "<<cnt<<" ";
            tmp_vec.push_back(Itval(s_vid, e_vid, minS, maxS, cnt));
        }
//        cout<<endl;
    }
    fin.close();
    cout<<"*** finish load_index_GRL2 ***"<<endl;
     */
}

void load_index_GRL3(string graph_name)
{
//    cout<<" *** in load_index_GRL3 ***"<<endl;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_GRL3.bin");

    cout<<"index name : "<<graph_name<<endl;
    
    FILE * f = Utility::open_file(graph_name.c_str(), "rb");
    vsn.resize(n);
    
    int vid, numofI;
    ui s_vid, e_vid;
    char minS, maxS;
    int cnt;
    unsigned long result;
    
    while(fread(&vid, sizeof(int), 1, f) == 1) {
//        cout<<"vid = "<<vid<<endl;
        fread(&numofI, sizeof(int), 1, f);
//        cout<<"numofI = "<<numofI<<endl;
//        cout.flush();
        vector<Itval> & tmp_vec = vsn[vid];
        if(numofI > 0){
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&e_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                fread(&maxS, sizeof(char), 1, f);
                fread(&cnt, sizeof(int), 1, f);
                tmp_vec.push_back(Itval(s_vid, e_vid, (double)minS/100, (double)maxS/100, cnt));
//                cout<<"   +  "<<s_vid<<","<<e_vid<<","<<(double)minS/100<<","<<(double)maxS/100<<","<<cnt<<endl;
                -- numofI;
                if(numofI == 0) break;
            }
        }
        else { //indi
            numofI = -numofI;
            while (1) {
                fread(&s_vid, sizeof(ui), 1, f);
                fread(&minS, sizeof(char), 1, f);
                
                tmp_vec.push_back(Itval(s_vid, s_vid, (double)minS/100, (double)minS/100, 1));
//                cout<<"   -  "<<s_vid<<","<<(double)minS/100<<","<<endl;
                
                -- numofI;
                if(numofI == 0) break;
            }
        }
    }
    fclose(f);
//    cout<<"*** finish load_index_GRL3.bin ***"<<endl;
    
    //txt version
    /*
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL3.txt");
    ifstream fin;
    fin.open(graph_name);
    if(!fin.is_open()){
        cout<<"cannot open index GRL3."<<endl; exit(1);
    }
    
    vsn.resize(n);
    
    string line;
    ui vid, s_vid, e_vid, cnt;
    double minS, maxS;
    while (getline(fin, line)) {
        stringstream ss(line);
        ss >> vid;
        assert(vid >=0 && vid < n);
        vector<Itval> & tmp_vec = vsn[vid];
//        cout<<" a new line, "<<vid<<": ";
        while (ss >> s_vid >> e_vid >> minS >> maxS >> cnt) {
//            cout<<s_vid<<", "<<e_vid<<", "<<minS<<", "<<maxS<<", "<<cnt<<" ";
            tmp_vec.push_back(Itval(s_vid, e_vid, minS, maxS, cnt));
        }
//        cout<<endl;
    }
    fin.close();
    cout<<"*** finish load_index_GRL3 ***"<<endl;
     */
}

inline bool jsub(ui & u, ui & v)
{
    for(auto & e : vsn[v]){
        if(u >= e.s_idx && u <= e.e_idx && e.max_score >= epsilon) return true;
        if(e.s_idx > u) break;
    }
    return false;
}

inline double jsvec(vector<ui> & vec1, vector<ui> & vec2)
{
    int du = vec1.size(), dv = vec2.size();
    ui idx1 = 0, idx2 = 0;
    ui idx1_end = idx1 + du, idx2_end = idx2 + dv;
    double common = 0;
//    int r1 = du, r2 = dv, dif = 0;
    
    while (idx1 < idx1_end && idx2 < idx2_end) {
        if(vec1[idx1] == vec2[idx2]){
            ++ common; ++ idx1; ++ idx2;
//            -- r1; -- r2;
        }
        else{
            if(vec1[idx1] > vec2[idx2]) {
                ++ idx2;
//                -- r2; ++ dif;
            }
            else {
                ++ idx1;
//                -- r1; ++ dif;
            }
        }
//        double ts = common + min(r1, r2);
//        if(ts / (dif + common + max(r1, r2)) < epsilon) return 0;
    }
//    cout<<"js("<<u<<","<<v<<"), common = "<<common<<", cardinality = "<<du + dv - common<<", js = "<<common/(du + dv - common)<<endl;
    return common/(du + dv - common);
}

inline double js(ui & u, ui & v)
{
    if(u == v) return 0;
    int du = degree[u], dv = degree[v];
    if( (double) min(du, dv) / max(du, dv) < epsilon) return 0;
    
//    if(LRnid[u].first < LRnid[v].first && LRnid[u].second > LRnid[v].first && LRnid[u].second < LRnid[v].second ){
//        int max_cn = LRnid[u].second - LRnid[v].first;
//        if( (double) max_cn / max(du, dv) < epsilon ) return 0;
//
//    }
//    else{ //LRnid[u].first >= LRnid[v].first
//        if(LRnid[u].first < LRnid[v].second && LRnid[u].second > LRnid[v].second){
//            int max_cn = LRnid[v].second - LRnid[u].first;
//            if( (double) max_cn / max(du, dv) < epsilon ) return 0;
//        }
//    }
    
    ui idx1 = pstart[u], idx2 = pstart[v];
    ui idx1_end = idx1 + du, idx2_end = idx2 + dv;
    double common = 0;
//    int r1 = du, r2 = dv, dif = 0;
    
    while (idx1 < idx1_end && idx2 < idx2_end) {
        if(edges[idx1] == edges[idx2]){
            ++ common; ++ idx1; ++ idx2;
//            -- r1; -- r2;
        }
        else{
            if(edges[idx1] > edges[idx2]) {
                ++ idx2;
//                -- r2; ++ dif;
            }
            else {
                ++ idx1;
//                -- r1; ++ dif;
            }
        }
//        double ts = common + min(r1, r2);
//        if(ts / (dif + common + max(r1, r2)) < epsilon) return 0;
    }
//    cout<<"js("<<u<<","<<v<<"), common = "<<common<<", cardinality = "<<du + dv - common<<", js = "<<common/(du + dv - common)<<endl;
    return common/(du + dv - common);
}

//按顺序扫描同侧所有点来获得与u相似度高于epsilon的点
vector<ui>  get_sim_nei(ui u)
{
    assert(u >= 0 && u < n);
    vector<ui> sim_nei;
    if(u < n1){ //u is from U
        for(ui i = 0; i < n1; i++){
            if(i == u) continue;
            if(js(u,i) >= epsilon) sim_nei.push_back(i);
        }
    }
    else{ // u is from V
        for(ui i = n1; i < n; i++){
            if(i == u) continue;
            if(js(u,i) >= epsilon) sim_nei.push_back(i);
        }
    }
    return sim_nei;
}

//计数u的2-hop neighbors来获得与u有至少一个com nei的点，存在may_sim_list中，UV_cnt存着每个点与u的com nei数目。
vector<ui> get_sim_nei2(ui u)
{
//    cout<<"now we are getting vertex "<<u<<" sim nei."<<endl;
    assert(u >= 0 && u < n);
    
    vector<ui> sim_nei;
    ui * UV_cnt = new ui[n]; //记录2-hopneighbor的count
    memset(UV_cnt, 0, sizeof(ui)*n);
    vector<ui> may_sim_list;
    
    for(ui i = pstart[u]; i < pstart[u+1]; i++){
        ui v = edges[i];
//        cout<<"    check its nei "<<v<<", degree = "<<degree[v]<<endl;
        for(ui j = pstart[v]; j < pstart[v+1]; j++){
            ui w = edges[j];
            if(w == u) continue;
//            cout<<"        see its nei nei "<<w<<endl;
            if(UV_cnt[w] == 0) {
                may_sim_list.push_back(w);
//                cout<<"1st time to see it put it into may_sim_list."<<endl;
            }
            ++ UV_cnt[w];
//            cout<<"++ its UV_cnt["<<w<<"] = "<<UV_cnt[w]<<endl;
        }
    }
//    cout<<"mat_sim_list size = "<<may_sim_list.size()<<endl;
//    for(auto e : may_sim_list) cout<<e<<", "; cout<<endl;
    
    for(auto e : may_sim_list){
        
        if(del_ver[e]) continue;
        
        assert(degree[e] >= UV_cnt[e]);
        assert(degree[u] >= UV_cnt[e]);
        double sim_score = (double) UV_cnt[e] / (degree[u] + degree[e] - UV_cnt[e]);
//        cout<<"for vertex "<<e<<" in mat_sim_list, its degree = "<<degree[e]<<", UN_cnt = "<<UV_cnt[e]<<", thus, sim score = "<<sim_score<<endl;
        if(sim_score >= epsilon) sim_nei.push_back(e);
        
    }
    delete [] UV_cnt;
    return sim_nei;
}

vector<ui> get_sim_neif(ui u)
{
//    cout<<"now we are getting vertex "<<u<<" sim nei."<<endl;
    assert(u >= 0 && u < n);
    
    vector<ui> sim_nei;
    vector<ui> may_sim_list;
    
    for(ui i = pstart[u]; i < pstart[u+1]; i++){
        ui v = edges[i];
//        cout<<"    check its nei "<<v<<", degree = "<<degree[v]<<endl;
        for(ui j = pstart[v]; j < pstart[v+1]; j++){
            ui w = edges[j];
            if(w == u) continue;
//            cout<<"        see its nei nei "<<w<<endl;
            if(comneicntforDBLP[w] == 0) {
                may_sim_list.push_back(w);
//                cout<<"1st time to see it put it into may_sim_list."<<endl;
            }
            ++ comneicntforDBLP[w];
//            cout<<"++ its UV_cnt["<<w<<"] = "<<UV_cnt[w]<<endl;
        }
    }
//    cout<<"mat_sim_list size = "<<may_sim_list.size()<<endl;
//    for(auto e : may_sim_list) cout<<e<<", "; cout<<endl;
    
    for(auto e : may_sim_list){
        
        if(del_ver[e]) continue;
        
        assert(degree[e] >= comneicntforDBLP[e]);
        assert(degree[u] >= comneicntforDBLP[e]);
        double sim_score = (double) comneicntforDBLP[e] / (degree[u] + degree[e] - comneicntforDBLP[e]);
//        cout<<"for vertex "<<e<<" in mat_sim_list, its degree = "<<degree[e]<<", UN_cnt = "<<UV_cnt[e]<<", thus, sim score = "<<sim_score<<endl;
        if(sim_score >= epsilon) sim_nei.push_back(e);
        
    }
    for(auto e : may_sim_list) comneicntforDBLP[e] = 0;
    return sim_nei;
}


void index_vr()
{
    cout<<"     *** index_vr ***"<<endl;
    if(vsn.empty()) {
        cout<<"vsn is empty."<<endl; exit(1);
    }
    assert(tau >= 2);
    
    Timer tt;
    int * tmp_deg = new int[n];
    queue<ui> Q;
    for(ui i = 0; i < n; i++) {
        tmp_deg[i] = degree[i];
        if(tmp_deg[i] < tau) Q.push(i);
    }

    int Q_init_size = Q.size();

    int delcnt = 0;
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1; delcnt++;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(tmp_deg[v]-- == tau) Q.push(v);
        }
    }
    int Q_extra_size = delcnt - Q_init_size;
//    long long left_e_cnt = 0;
//    for(ui i = 0; i < n; i++){
//        if(del_ver[i] == 0){
//            left_e_cnt += tmp_deg[i];
//        }
//    }
//    assert(left_e_cnt%2 == 0);
//    left_e_cnt /= 2;
//    cout<<"EDGE: deleted "<<m-left_e_cnt<<" (remaining "<<left_e_cnt<<")."<<endl;
    cout<<"     Phase 1, deleted "<<delcnt<<", ("<<Q_init_size<<"+"<<Q_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
    
    int * Sdeg = new int[n];
    for(ui u = 0; u < n; u++){
        if(del_ver[u]) continue;
        const vector<Itval> & cand_list = vsn[u];
        int tmp_sdeg = 0;
        for(auto & e : cand_list){
            if(e.max_score < epsilon) continue;
            for(ui i = e.s_idx; i <= e.e_idx; i++){
                if(!del_ver[i] && js(u, i) >= epsilon && i != u) ++ tmp_sdeg;
            }
        }
        
        Sdeg[u] = tmp_sdeg;
        if(tmp_sdeg < tau - 1) {Q.push(u);}
    }
    
    int Q_phase2_init_size = Q.size();
    int phase2_total_del = 0;
    
//    cout<<"         Phase 2, init Q ("<<Q.size()<<") according to similar degree, time cost = "<<integer_to_string(tt.elapsed())<<endl;
    long long phase2_init_Q_timecost = tt.elapsed();
    tt.restart();
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1; ++ delcnt; ++ phase2_total_del;
        
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(del_ver[v]) continue;
            if(tmp_deg[v] -- == tau && Sdeg[v] >= tau - 1) Q.push(v);
        }
        
        vector<ui> simnei;
        const vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list){
            if(e.max_score < epsilon) continue;
            for(ui i = e.s_idx; i <= e.e_idx; i++){
                if(del_ver[i]) continue;
                if(js(u, i) >= epsilon && i != u) simnei.push_back(i);
            }
        }
        for(ui w : simnei){
            if(Sdeg[w] -- == tau - 1 && tmp_deg[w] >= tau) Q.push(w);
        }
    }
    
    int Q_phase2_extra_size = phase2_total_del - Q_phase2_init_size;
    
//    cout<<"         Phase 2, clear Q to empty, time cost = "<<integer_to_string(tt.elapsed())<<endl;
    long long phase2_clear_Q_timecost = tt.elapsed();
    cout<<"     Phase 2, deleted "<<Q_phase2_init_size+Q_phase2_extra_size<<", ("<<Q_phase2_init_size<<"+"<<Q_phase2_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(phase2_init_Q_timecost)<<" + "<<integer_to_string(phase2_clear_Q_timecost)<<endl;
    delete [] tmp_deg;
    delete [] Sdeg;
    
    cout<<"     Deleted "<<delcnt<<" vertices. (remaining "<<n-delcnt<<")."<<endl;
}

void index_vr_opt()
{
    cout<<"     *** index_vr_opt ***"<<endl;
    if(vsn.empty()) {
        cout<<"vsn is empty."<<endl; exit(1);
    }
    assert(tau >= 2);
    
    Timer tt;
    queue<ui> Q;
    for(ui i = 0; i < n; i++) if(TMPdeg[i] < tau) Q.push(i);
    
    int Q_init_size = Q.size();
    int delcnt = 0;
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1;
        ++ delcnt;
        for(ui i = pstart[u]; i < pstart[u+1]; i++) if(TMPdeg[edges[i]]-- == tau) Q.push(edges[i]);
    }
    int Q_extra_size = delcnt - Q_init_size;

    cout<<"     Phase 1, deleted "<<delcnt<<", ("<<Q_init_size<<"+"<<Q_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
    
    int sd;
    for(ui u = 0; u < n; u++) if(!del_ver[u]) {
        const vector<Itval> & cand_list = vsn[u];
        sd = 0;
        for(auto & e : cand_list) if(e.max_score >= epsilon) sd += e.c;
        if(sd < tau - 1) {
            Q.push(u);
            del_ver[u] = 1;  //这一步 del_ver[] = 1 表示进Q了
        }
    }
    
    int Q_phase15_init_size = Q.size();
    int phase15_total_del = 0;
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        ++ phase15_total_del; ++ delcnt;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(TMPdeg[v]-- == tau && !del_ver[v]) {
                Q.push(v);
                del_ver[v] = 1;
            }
        }
    }
    assert(phase15_total_del >= Q_phase15_init_size);
    int Q_phase15_extra_size = phase15_total_del - Q_phase15_init_size;
    cout<<"     Phase15, deleted "<<phase15_total_del<<", ("<<Q_phase15_init_size<<"+"<<Q_phase15_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
        
    //到目前为止，TMPdeg[] 应该还是正确的
    
    for(ui u = 0; u < n; u++) if(!del_ver[u]) {
        assert(TMPdeg[u] >= tau);
        vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list) if(e.max_score >= epsilon) {
#ifdef _SHRINK_
            bool f = false;
#endif
//            cout<<"this e : "<<e.s_idx<<" , "<<e.e_idx<<" , "<<e.min_score<<" , "<<e.max_score<<" , "<<e.c<<endl;
            for(ui i = e.s_idx; i <= e.e_idx; i++) if(!del_ver[i] && js(u, i) >= epsilon && i != u) {
                ++ Sdeg[u];
#ifdef _SHRINK_
                f = true;
#endif
            }
#ifdef _SHRINK_
            if(!f) {
                e.max_score = 0;
                e.min_score = 0;
                e.c = 0;
//                cout<<"     we change e : "<<e.s_idx<<" , "<<e.e_idx<<" , "<<e.min_score<<" , "<<e.max_score<<" , "<<e.c<<endl;
            }
#endif
        }
        if(Sdeg[u] < tau - 1) Q.push(u);
    }
    
//另一种方式，更快只算 i > u
//    for(ui u = 0; u < n; u++) if(!del_ver[u]) {
//        assert(TMPdeg[u] >= tau);
//        const vector<Itval> & cand_list = vsn[u];
//        for(auto & e : cand_list) if(e.max_score >= epsilon) {
//            for(ui i = e.s_idx; i <= e.e_idx; i++) if(!del_ver[i] && i > u && js(u, i) >= epsilon) {
//                ++ Sdeg[u]; ++ Sdeg[i];
//            }
//        }
//    }
//    for(ui u = 0; u < n; u++) if(!del_ver[u] && Sdeg[u] < tau - 1) Q.push(u);
    
    int Q_phase2_init_size = Q.size();
    int phase2_total_del = 0;
    
    long long phase2_init_Q_timecost = tt.elapsed();
    tt.restart();

    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1; ++ delcnt; ++ phase2_total_del;
        
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(del_ver[v]) continue;
            if(TMPdeg[v] -- == tau && Sdeg[v] >= tau - 1) {
                Q.push(v);
            }
        }
        
        vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list){
            if(e.max_score < epsilon) continue;
#ifdef _SHRINK_
            bool f = false;
#endif
            for(ui i = e.s_idx; i <= e.e_idx; i++){
                if(del_ver[i]) continue;
                if(js(u, i) >= epsilon && i != u) {
                    if(Sdeg[i] -- == tau - 1 && TMPdeg[i] >= tau) {
                        Q.push(i);
                    }
#ifdef _SHRINK_
                    f = true;
#endif
                }
            }
#ifdef _SHRINK_
            if(!f) {
                e.max_score = 0;
                e.min_score = 0;
                e.c = 0;
            }
#endif
        }
    }
    
    //到现在，剩余点 (i.e., del_ver[] = 0 ) 的 TMdeg[] 和 Sdeg[] 应该还是对的
    
    int Q_phase2_extra_size = phase2_total_del - Q_phase2_init_size;
    
//    cout<<"         Phase 2, clear Q to empty, time cost = "<<integer_to_string(tt.elapsed())<<endl;
    long long phase2_clear_Q_timecost = tt.elapsed();
    cout<<"     Phase 2, deleted "<<Q_phase2_init_size+Q_phase2_extra_size<<", ("<<Q_phase2_init_size<<"+"<<Q_phase2_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(phase2_init_Q_timecost)<<" + "<<integer_to_string(phase2_clear_Q_timecost)<<endl;
    cout<<"     Deleted "<<delcnt<<" vertices. (remaining "<<n-delcnt<<")."<<endl;
    
}

void index_vr_opt_noRSim()
{
//    cout<<"     *** index_vr_opt ***"<<endl;
    if(vsn.empty()) {
        cout<<"vsn is empty."<<endl; exit(1);
    }
    assert(tau >= 2);
    
    Timer tt;
    queue<ui> Q;
    for(ui i = 0; i < n; i++) if(TMPdeg[i] < tau) Q.push(i);
    
    int Q_init_size = Q.size();
    int delcnt = 0;
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1;
        ++ delcnt;
        for(ui i = pstart[u]; i < pstart[u+1]; i++) if(TMPdeg[edges[i]]-- == tau) Q.push(edges[i]);
    }
    int Q_extra_size = delcnt - Q_init_size;

//    cout<<"     Phase 1, deleted "<<delcnt<<", ("<<Q_init_size<<"+"<<Q_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
    
    
    assert(noRSim == 1 || noRSim == 2);
    int sd;
    for(ui u = 0; u < n1; u++) if(!del_ver[u]) {
        const vector<Itval> & cand_list = vsn[u];
        sd = 0;
        for(auto & e : cand_list) if(e.max_score >= epsilon) sd += e.c;
        if(sd < tau - 1) {
            Q.push(u);
            del_ver[u] = 1;  //这一步 del_ver[] = 1 表示进Q了
        }
    }
    
    int Q_phase15_init_size = Q.size();
    int phase15_total_del = 0;
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        ++ phase15_total_del; ++ delcnt;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(TMPdeg[v]-- == tau && !del_ver[v]) {
                Q.push(v);
                del_ver[v] = 1;
            }
        }
    }
    assert(phase15_total_del >= Q_phase15_init_size);
    int Q_phase15_extra_size = phase15_total_del - Q_phase15_init_size;
//    cout<<"     Phase15, deleted "<<phase15_total_del<<", ("<<Q_phase15_init_size<<"+"<<Q_phase15_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
        
    //到目前为止，TMPdeg[] 应该还是正确的
    for(ui u = 0; u < n1; u++) if(!del_ver[u]) {
        assert(TMPdeg[u] >= tau);
        vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list) if(e.max_score >= epsilon) {
//            cout<<"this e : "<<e.s_idx<<" , "<<e.e_idx<<" , "<<e.min_score<<" , "<<e.max_score<<" , "<<e.c<<endl;
            for(ui i = e.s_idx; i <= e.e_idx; i++) if(!del_ver[i] && js(u, i) >= epsilon && i != u) {
                ++ Sdeg[u];
            }
        }
        if(Sdeg[u] < tau - 1) Q.push(u);
    }
    
    int num_of_vertice_inR = 0;
    for(ui u = n1; u < n; u++) if(!del_ver[u]) ++ num_of_vertice_inR;
    -- num_of_vertice_inR;
    for(ui u = n1; u < n; u++) if(!del_ver[u]) Sdeg[u] = num_of_vertice_inR;
    
    int Q_phase2_init_size = Q.size();
    int phase2_total_del = 0;
    
    long long phase2_init_Q_timecost = tt.elapsed();
    tt.restart();

    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1; ++ delcnt; ++ phase2_total_del;
        
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(del_ver[v]) continue;
            if(TMPdeg[v] -- == tau && Sdeg[v] >= tau - 1) {
                Q.push(v);
            }
        }
        
        if(os[u]==2) continue;
        
        vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list){
            if(e.max_score < epsilon) continue;
            for(ui i = e.s_idx; i <= e.e_idx; i++){
                if(del_ver[i]) continue;
                if(js(u, i) >= epsilon && i != u) {
                    if(Sdeg[i] -- == tau - 1 && TMPdeg[i] >= tau) {
                        Q.push(i);
                    }
                }
            }
        }
    }
    
    num_of_vertice_inR = 0;
    for(ui u = n1; u < n; u++) if(!del_ver[u]) ++ num_of_vertice_inR;
    -- num_of_vertice_inR;
    for(ui u = n1; u < n; u++) if(!del_ver[u]) Sdeg[u] = num_of_vertice_inR;
    //到现在，剩余点 (i.e., del_ver[] = 0 ) 的 TMdeg[] 和 Sdeg[] 应该还是对的
    
    int Q_phase2_extra_size = phase2_total_del - Q_phase2_init_size;
    
//    cout<<"         Phase 2, clear Q to empty, time cost = "<<integer_to_string(tt.elapsed())<<endl;
    long long phase2_clear_Q_timecost = tt.elapsed();
//    cout<<"     Phase 2, deleted "<<Q_phase2_init_size+Q_phase2_extra_size<<", ("<<Q_phase2_init_size<<"+"<<Q_phase2_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(phase2_init_Q_timecost)<<" + "<<integer_to_string(phase2_clear_Q_timecost)<<endl;
    cout<<"     Deleted "<<delcnt<<" vertices. (remaining "<<n-delcnt<<")."<<endl;
        
    
    //***
//    vector<ui> Cll;
//    for(ui u = 0; u < n1; u++) if(!del_ver[u]) Cll.push_back(u);
//    vector<ui> Crr;
//    for(ui u = n1; u < n; u++) if(!del_ver[u]) Crr.push_back(u);
//    assert(results.size()==0);
//    results.push_back(make_pair(Cll, Crr));
}


void index_vr_opt2()
{
    cout<<"     *** index_vr_opt2 ***"<<endl;
    if(vsn.empty()) {
        cout<<"vsn is empty."<<endl; exit(1);
    }
    assert(tau >= 2);
    
    Timer tt;
    queue<ui> Q;
    for(ui i = 0; i < n; i++) if(TMPdeg[i] < tau) Q.push(i);
    
    int Q_init_size = Q.size();
    int delcnt = 0;
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1;
        ++ delcnt;
        for(ui i = pstart[u]; i < pstart[u+1]; i++) if(TMPdeg[edges[i]]-- == tau) Q.push(edges[i]);
    }
    int Q_extra_size = delcnt - Q_init_size;

    cout<<"     Phase 1, deleted "<<delcnt<<", ("<<Q_init_size<<"+"<<Q_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
    
    int sd;
    for(ui u = 0; u < n; u++) if(!del_ver[u]) {
        const vector<Itval> & cand_list = vsn[u];
        sd = 0;
        for(auto & e : cand_list) if(e.max_score >= epsilon) sd += e.c;
        Sdeg[u] = sd;
        if(sd < tau - 1) {
            Q.push(u);
            del_ver[u] = 1;
        }
    }
    
    int Q_phase15_init_size = Q.size();
    int phase15_total_del = 0;
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        ++ phase15_total_del; ++ delcnt;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(TMPdeg[v]-- == tau && !del_ver[v]) {
                Q.push(v);
                del_ver[v] = 1;
            }
        }
//        vector<ui> simnei;
        const vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list) if(e.min_score >= epsilon && (e.e_idx - e.s_idx + 1) == e.c) {
            for(ui i = e.s_idx; i <= e.e_idx; i++){
                if(Sdeg[i] -- == tau - 1 && !del_ver[i]){
                    Q.push(i);
                    del_ver[i] = 1;
                }
            }
        }
    }
    assert(phase15_total_del >= Q_phase15_init_size);
    int Q_phase15_extra_size = phase15_total_del - Q_phase15_init_size;
    cout<<"     Phase15, deleted "<<phase15_total_del<<", ("<<Q_phase15_init_size<<"+"<<Q_phase15_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(tt.elapsed())<<endl;
    tt.restart();
    
//    for(ui u = 0; u < n; u++) if(!del_ver[u]) {
//        assert(TMPdeg[u] >= tau);
//        const vector<Itval> & cand_list = vsn[u];
//        sd = 0;
//        for(auto & e : cand_list) if(e.max_score >= epsilon) {
//            for(ui i = e.s_idx; i <= e.e_idx; i++) if(!del_ver[i] && js(u, i) >= epsilon && i != u)
//                ++ sd;
//        }
//        Sdeg[u] = sd;
//        if(sd < tau - 1) {Q.push(u);}
//    }
    
    for(ui u = 0; u < n; u++) if(!del_ver[u]) Sdeg[u] = 0;
    for(ui u = 0; u < n; u++) if(!del_ver[u]) {
        assert(TMPdeg[u] >= tau);
        const vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list) if(e.max_score >= epsilon) {
            for(ui i = e.s_idx; i <= e.e_idx; i++) if(!del_ver[i] && i > u && js(u, i) >= epsilon)
            {
                ++ Sdeg[u]; ++ Sdeg[i];
            }
        }
    }
    for(ui u = 0; u < n; u++) if(!del_ver[u] && Sdeg[u] < tau - 1) Q.push(u);
    
    int Q_phase2_init_size = Q.size();
    int phase2_total_del = 0;

    long long phase2_init_Q_timecost = tt.elapsed();
    tt.restart();
    
    while (!Q.empty()) {
        ui u = Q.front(); Q.pop();
        del_ver[u] = 1; ++ delcnt; ++ phase2_total_del;
        
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            if(del_ver[v]) continue;
            if(TMPdeg[v] -- == tau && Sdeg[v] >= tau - 1) Q.push(v);
        }
        
        const vector<Itval> & cand_list = vsn[u];
        for(auto & e : cand_list){
            if(e.max_score < epsilon) continue;
            for(ui i = e.s_idx; i <= e.e_idx; i++){
                if(del_ver[i]) continue;
                if(js(u, i) >= epsilon && i != u) {
                    if(Sdeg[i] -- == tau - 1 && TMPdeg[i] >= tau) Q.push(i);
                }
            }
        }
    }
    
    int Q_phase2_extra_size = phase2_total_del - Q_phase2_init_size;
    
    long long phase2_clear_Q_timecost = tt.elapsed();
    cout<<"     Phase 2, deleted "<<Q_phase2_init_size+Q_phase2_extra_size<<", ("<<Q_phase2_init_size<<"+"<<Q_phase2_extra_size<<", remain "<<n-delcnt<<"), time cost = "<<integer_to_string(phase2_init_Q_timecost)<<" + "<<integer_to_string(phase2_clear_Q_timecost)<<endl;
    cout<<"     Deleted "<<delcnt<<" vertices. (remaining "<<n-delcnt<<")."<<endl;
}

void init_hash()
{
    sup = new dense_hash_map<ui, int>[n1];
    edel = new dense_hash_map<ui, bool>[n1];
    esign = new dense_hash_map<ui, bool>[n1];
    for(ui i = 0; i < n1; i++){
        sup[i].set_empty_key(INF);
        edel[i].set_empty_key(INF);
        esign[i].set_empty_key(INF);
    }
    for(ui i = 0; i < n1; i++){
        for(ui j = pstart[i]; j < pstart[i+1]; j++){
            ui v = edges[j];
            sup[i][v] = 0;
            edel[i][v] = 0;
            esign[i][v] = 0;
        }
    }
}


void obtain_degree_inP(vector<ui> &PL, vector<ui> &PR)
{
//    cout<<"\t ** in obtain_degree_inP()."<<endl;
    for(auto e : PL) {
        pdeg_inP[e] = 0;
        ndeg_inP[e] = 0;
//        deg_inP[e] = 0;
    }
    for(auto e : PR) {
        pdeg_inP[e] = 0;
        ndeg_inP[e] = 0;
//        deg_inP[e] = 0;
    }
    
    for(ui i = 0; i < PL.size(); i++){
        for(ui j = i + 1; j < PL.size(); j++){
            ui u = PL[i], v = PL[j];
            if(Matrix[trans[u]][trans[v]] != 0){
//                ++ deg_inP[u]; ++ deg_inP[v];
                ++ pdeg_inP[u]; ++ pdeg_inP[v];
            }
        }
    }
    for(ui i = 0; i < PR.size(); i++){
        for(ui j = i + 1; j < PR.size(); j++){
            ui u = PR[i], v = PR[j];
            if(Matrix[trans[u]][trans[v]] != 0){
//                ++ deg_inP[u]; ++ deg_inP[v];
                ++ pdeg_inP[u]; ++ pdeg_inP[v];
            }
        }
    }
    for(auto u : PL){
        for(auto v : PR){
            if(Matrix[trans[u]][trans[v]] != 0){
//                ++ deg_inP[u]; ++ deg_inP[v];
                ++ ndeg_inP[u]; ++ ndeg_inP[v];
            }
        }
    }
    
//    cout<<"         for each vertex in PL : "<<endl;
//    for(auto e : PL){
//        cout<<"\t           "<<e<<" : deg_inP = "<<deg_inP[e]<<", pdeg_inP = "<<pdeg_inP[e]<<", ndeg_inP = "<<ndeg_inP[e]<<endl;
//    }
//    cout<<"         for each vertex in PR : "<<endl;
//    for(auto e : PR){
//        cout<<"\t           "<<e<<" : deg_inP = "<<deg_inP[e]<<", pdeg_inP = "<<pdeg_inP[e]<<", ndeg_inP = "<<ndeg_inP[e]<<endl;
//    }
}

void pruneP_by_degree(vector<ui> &CL, vector<ui> &CR, vector<ui> &PL, vector<ui> &PR)
{
//    cout<<"in pruneP_by_degree()."<<endl;
    int L_pt = tau - (int)CL.size() - 1;
    int L_nt = tau - (int)CR.size();
    int R_pt = tau - (int)CR.size() - 1;
    int R_nt = tau - (int)CL.size();
//    cout<<"L_pt = "<<L_pt<<", L_nt = "<<L_nt<<", R_pt = "<<R_pt<<", R_nt = "<<R_nt<<endl;
    if(L_pt <= 0 && L_nt <= 0 && R_pt <= 0 && R_nt <= 0) return;
    queue<ui> Q;
    for(auto e : PL) if(pdeg_inP[e] < L_pt || ndeg_inP[e] < L_nt) {
        Q.push(e);
//        cout<<"put "<<e<<" into Q."<<endl;
    }
    for(auto e : PR) if(pdeg_inP[e] < R_pt || ndeg_inP[e] < R_nt) {
        Q.push(e);
//        cout<<"put "<<e<<" into Q."<<endl;
    }
    
    ui * x = new ui[r_n];
    memset(x, 0, sizeof(ui) * r_n);
    
    while (!Q.empty()) {
        ui u = Q.front();
//        cout<<"     pop "<<u<<endl;
        Q.pop();
        x[u] = 1;
        for(auto v : PL){
            if(Matrix[trans[u]][trans[v]] == 1){
//                cout<<"         will decrease its p nei deg "<<v<<endl;
                assert(pdeg_inP[v] > 0);
                if(pdeg_inP[v] -- == L_pt && ndeg_inP[v] >= L_nt) {
                    Q.push(v);
//                    cout<<"             put "<<v<<" into Q."<<endl;
                }
            }
            else if (Matrix[trans[u]][trans[v]] == -1){
//                cout<<"         will decrease its n nei deg "<<v<<endl;
                assert(ndeg_inP[v] > 0);
                if(ndeg_inP[v] -- == L_nt && pdeg_inP[v] >= L_pt) {
                    Q.push(v);
//                    cout<<"             put "<<v<<" into Q."<<endl;
                }
            }
        }
        for(auto v : PR){
            if(Matrix[trans[u]][trans[v]] == 1){
//                cout<<"         will decrease its p nei deg "<<v<<endl;
                assert(pdeg_inP[v] > 0);
                if(pdeg_inP[v] -- == R_pt && ndeg_inP[v] >= R_nt) {
                    Q.push(v);
//                    cout<<"             put "<<v<<" into Q."<<endl;
                }
            }
            else if (Matrix[trans[u]][trans[v]] == -1){
//                cout<<"         will decrease its n nei deg "<<v<<endl;
                assert(ndeg_inP[v] > 0);
                if(ndeg_inP[v] -- == R_nt && pdeg_inP[v] >= R_pt) {
                    Q.push(v);
//                    cout<<"             put "<<v<<" into Q."<<endl;
                }
            }
        }
    }
    vector<ui> newPL, newPR;
    for(auto e : PL) if(x[e] == 0) newPL.push_back(e);
    for(auto e : PR) if(x[e] == 0) newPR.push_back(e);
    PL = newPL;
    PR = newPR;
    delete [] x;
    x = nullptr;
//    cout<<"finally : "<<endl;
//    cout<<"PL : ";
//    for(auto e : PL) cout<<e<<" , "; cout<<endl;
//    cout<<"PR : ";
//    for(auto e : PR) cout<<e<<" , "; cout<<endl;
}

ui pivot_selection(vector<ui> &PL, vector<ui> &PR, vector<ui> &QL, vector<ui> &QR)
{
    for(auto e : PL) deg_inP[e] = pdeg_inP[e] + ndeg_inP[e];
    for(auto e : PR) deg_inP[e] = pdeg_inP[e] + ndeg_inP[e];
    
    for(auto e : QL) deg_inP[e] = 0;
    for(auto e : QR) deg_inP[e] = 0;
    for(auto u : QL){
        for(auto v : PL) if(Matrix[trans[u]][trans[v]] != 0) ++ deg_inP[u];
        for(auto v : PR) if(Matrix[trans[u]][trans[v]] != 0) ++ deg_inP[u];
    }
    for(auto u : QR){
        for(auto v : PL) if(Matrix[trans[u]][trans[v]] != 0) ++ deg_inP[u];
        for(auto v : PR) if(Matrix[trans[u]][trans[v]] != 0) ++ deg_inP[u];
    }
    
    ui pivot = 0;
    int max_local_deg = -1;
    for(auto e : PL) if(deg_inP[e] > max_local_deg) {max_local_deg = deg_inP[e]; pivot = e; }
    for(auto e : PR) if(deg_inP[e] > max_local_deg) {max_local_deg = deg_inP[e]; pivot = e; }
    for(auto e : QL) if(deg_inP[e] > max_local_deg) {max_local_deg = deg_inP[e]; pivot = e; }
    for(auto e : QR) if(deg_inP[e] > max_local_deg) {max_local_deg = deg_inP[e]; pivot = e; }
    
    assert(max_local_deg >= 0);
    
    return pivot;
//    cout<<" deg inP : "<<endl;
//    cout<<"\t PL : "<<endl;
//    for(auto e : PL) cout<<"\t "<<e<<" deg = "<<deg_inP[e]<<endl;
//    cout<<"\t PR : "<<endl;
//    for(auto e : PR) cout<<"\t "<<e<<" deg = "<<deg_inP[e]<<endl;
//    cout<<"\t QL : "<<endl;
//    for(auto e : QL) cout<<"\t "<<e<<" deg = "<<deg_inP[e]<<endl;
//    cout<<"\t QR : "<<endl;
//    for(auto e : QR) cout<<"\t "<<e<<" deg = "<<deg_inP[e]<<endl;
}

void matrix_based_enum_core(vector<ui> CL, vector<ui> CR, vector<ui> PL, vector<ui> PR, vector<ui> QL, vector<ui> QR)
{
//    cout<<"\t ****** in a new ENUM round : "<<endl;
//    cout<<"\t CL : "; for (auto e : CL) cout<<e<<", "; cout<<endl;
//    cout<<"\t CR : "; for (auto e : CR) cout<<e<<", "; cout<<endl;
//    cout<<"\t PL : "; for (auto e : PL) cout<<e<<", "; cout<<endl;
//    cout<<"\t PR : "; for (auto e : PR) cout<<e<<", "; cout<<endl;
//    cout<<"\t QL : "; for (auto e : QL) cout<<e<<", "; cout<<endl;
//    cout<<"\t QR : "; for (auto e : QR) cout<<e<<", "; cout<<endl;
    
    if(over_time_flag) return;
    if(((double)clock() / CLOCKS_PER_SEC - startTime) > maxTime) over_time_flag = true;
    
    if(PL.empty() && PR.empty()){
        if(QL.empty() && QR.empty()){
            if(CL.size() >= tau && CR.size() >= tau){
#ifdef _CheckResults_
                vector<ui> C1, C2;
                for(auto e : CL) C1.push_back(oid[e]);
                for(auto e : CR) C2.push_back(oid[e]);
                sort(C1.begin(), C1.end(), less<>());
                sort(C2.begin(), C2.end(), less<>());
                if(C1[0] < C2[0]) results.push_back(make_pair(C1, C2));
                else results.push_back(make_pair(C2, C1));
#endif
                
                ++ MDBC_num;
                if(CL.size() + CR.size() > max_MDBC_size) max_MDBC_size = CL.size() + CR.size();
                if(CL.size() + CR.size() < min_MDBC_size) min_MDBC_size = CL.size() + CR.size();
            }
        }
        return;
    }
    if(CL.size() + PL.size() < tau || CR.size() + PR.size() < tau) return;
    obtain_degree_inP(PL, PR);  //validate pdeg_inP[] and ndeg_inP[], which will be use for P pruning.
    pruneP_by_degree(CL, CR, PL, PR);
    if(PL.empty() && PR.empty()) return;
    if(CL.size() + PL.size() < tau || CR.size() + PR.size() < tau) return;
    
    ui pivot = pivot_selection(PL, PR, QL, QR);  //validate deg_inP[]
    
    sort(PL.begin(), PL.end(), cmp_degP);
    sort(PR.begin(), PR.end(), cmp_degP);
    
    ui PLsize = (ui)PL.size();
    ui PRsize = (ui)PR.size();
    
    vector<int> PL_pivot(PLsize, 1);
    vector<int> PR_pivot(PRsize, 1);
    for(ui i = 0; i < PL.size(); i++) if(Matrix[trans[PL[i]]][trans[pivot]] != 0) PL_pivot[i] = 0;
    for(ui i = 0; i < PR.size(); i++) if(Matrix[trans[PR[i]]][trans[pivot]] != 0) PR_pivot[i] = 0;
    
    vector<int> PL_exp(PLsize, 1);
    vector<int> PR_exp(PRsize, 1);
    
    for(ui i = 0; i < PL.size(); i++){
        
        if(PL_pivot[i] == 0) continue;
        
        ui u = PL[i];
        
        vector<ui> newCL = CL;
        newCL.push_back(u);
        
        vector<ui> newCR = CR;
        
        vector<ui> newPL;
        for(ui j = 0; j < PL.size(); j++) if(Matrix[trans[PL[j]]][trans[u]] != 0 && PL_exp[j] != 0)
            newPL.push_back(PL[j]);
        
        vector<ui> newPR;
        for(ui j = 0; j < PR.size(); j++) if(Matrix[trans[PR[j]]][trans[u]] != 0)
            newPR.push_back(PR[j]);
        
        vector<ui> newQL;
        for(ui j = 0; j < QL.size(); j++) if(Matrix[trans[QL[j]]][trans[u]] != 0)
            newQL.push_back(QL[j]);
        
        vector<ui> newQR;
        for(ui j = 0; j < QR.size(); j++) if(Matrix[trans[QR[j]]][trans[u]] != 0)
            newQR.push_back(QR[j]);
        
        matrix_based_enum_core(newCR, newCL, newPR, newPL, newQR, newQL);
        
        PL_exp[i] = 0;
        QL.push_back(u);
    }
    
    for(ui i = 0; i < PR.size(); i++){
        
        if(PR_pivot[i] == 0) continue;
        
        ui u = PR[i];
        
        vector<ui> newCL = CL;
        
        vector<ui> newCR = CR;
        newCR.push_back(u);
        
        vector<ui> newPL;
        for(ui j = 0; j < PL.size(); j++) if(Matrix[trans[PL[j]]][trans[u]] != 0 && PL_exp[j] != 0)
            newPL.push_back(PL[j]);
        
        vector<ui> newPR;
        for(ui j = 0; j < PR.size(); j++) if(Matrix[trans[PR[j]]][trans[u]] != 0 && PR_exp[j] != 0)
            newPR.push_back(PR[j]);
        
        vector<ui> newQL;
        for(ui j = 0; j < QL.size(); j++) if(Matrix[trans[QL[j]]][trans[u]] != 0)
            newQL.push_back(QL[j]);
        
        vector<ui> newQR;
        for(ui j = 0; j < QR.size(); j++) if(Matrix[trans[QR[j]]][trans[u]] != 0)
            newQR.push_back(QR[j]);
        
        matrix_based_enum_core(newCR, newCL, newPR, newPL, newQR, newQL);
        
        PR_exp[i] = 0;
        QR.push_back(u);
    }
    
}

//.
void Enum_noRSim_adv_core(vector<ui> CL, vector<ui> CR, vector<ui> PL, vector<ui> QL)
{
//    cout<<"         === in a recursive invoke of Enum_noRSim_adv_core()"<<endl;
//    cout<<"         CL size = "<<CL.size()<<" : "; for(auto e : CL) cout<<e<<" ";cout<<endl;
//    cout<<"         CR size = "<<CR.size()<<" : "; for(auto e : CR) cout<<e<<" ";cout<<endl;
//    cout<<"         PL size = "<<PL.size()<<" : "; for(auto e : PL) cout<<e<<" ";cout<<endl;
//    cout<<"         QL size = "<<QL.size()<<" : "; for(auto e : QL) cout<<e<<" ";cout<<endl;
    
    if(over_time_flag) return;

    double DurTime = (double)clock() / CLOCKS_PER_SEC - each_start_time;
    
    Timer emT;

    if(DurTime > maxdurtime)
        over_time_flag = true;
    
    if(CL.size() + PL.size() < tau || CR.size() < tau) {
//        cout<<"         CL.size() + PL.size() < tau || CR.size() < tau, RETURN!!!"<<endl;
        return;
    }
    
    emT.restart();
    
    bool maximality = true;
    
    //for each vertex in PL and QL, obtain its degree in CR
//    cout<<"             $$$ for each vertex in PL and QL, obtain its degree in CR"<<endl;
    
    for(auto u : QL) {
        int d = 0;
        for(auto v : CR) {
            if(Matrix[trans[u]][trans[v]] == 1) {
                ++ d;
            }
        }
        if (d == CR.size()) {
            maximality = false;
            //early termination
#ifdef _ET_
            bool sim2PL = true; //check if u is similar to all vertices in PL
            for(auto e : PL) {
                if(Matrix[trans[u]][trans[e]] != 1) {
                    sim2PL = false;
                    break;
                }
            }
            if(sim2PL == true) {
                return;
            }
#endif
            //early termination
        }
        deg_inCR[u] = d;
    }
    
    for(auto u : PL) {
        int d = 0;
        for(auto v : CR) {
            if(Matrix[trans[u]][trans[v]] == 1) {
                ++ d;
            }
        }
        if (d == CR.size()) {
            maximality = false;
        }
        deg_inCR[u] = d;
    }
    
    enumT_Maximal_ET += emT.elapsed();
    emT.restart();
    
    if(maximality == true) {
        if(CL.size() >= tau && CR.size() >= tau) {
//            cout<<" @@@ find a result : < ";
//            for(auto e : CL) cout<<e<<" "; cout<<"| ";
//            for(auto e : CR) cout<<e<<" "; cout<<">";cout<<endl;

#ifdef _CheckResults_
                vector<ui> C1, C2;
                for(auto e : CL) C1.push_back(oid[e]);
                for(auto e : CR) C2.push_back(oid[e]);
                sort(C1.begin(), C1.end(), less<>());
                sort(C2.begin(), C2.end(), less<>());
                if(C1[0] < C2[0]) results.push_back(make_pair(C1, C2));
                else results.push_back(make_pair(C2, C1));
#endif
                
                ++ MDBC_num;
                if(CL.size() + CR.size() > max_MDBC_size) max_MDBC_size = CL.size() + CR.size();
                if(CL.size() + CR.size() < min_MDBC_size) min_MDBC_size = CL.size() + CR.size();
        }
    }
    
    if(PL.empty()) return;

    emT.restart();
#ifdef _ORDER_
    sort(PL.begin(), PL.end(), cmp_degCR);
#endif

    enumT_Sort += emT.elapsed();
    emT.restart();
    
    vector<ui> dom_cand;
    
#ifdef _DOM_
    //domination
    ui ustar;
    int dom_num = -1;
    
    vector<ui> mixvec;
    
    if(domway == 1) {
        mixvec = QL;
        for(auto e : PL) mixvec.push_back(e);
    }

    else{
        int tmpd = -1;
        ui tmpv;
        for(auto e : QL) if(deg_inCR[e] > tmpd) {
            tmpd = deg_inCR[e];
            tmpv = e;
        }
        for(auto e : PL) if(deg_inCR[e] > tmpd) {
            tmpd = deg_inCR[e];
            tmpv = e;
        }
        assert(tmpd >= 0);
        mixvec.push_back(tmpv);
    }
    
    for(auto u : mixvec) {
        vector<ui> dom_collection;
        //u's similar neighbors in QL and PL
        vector<ui> simnei;
        for(auto v : PL) if(Matrix[trans[u]][trans[v]] == 1) simnei.push_back(v);
        
        //u's connection neighbors in CR
        vector<ui> connei;
        for(auto v : CR) if(Matrix[trans[u]][trans[v]] == 1) {
            connei.push_back(v); domCR[v] = 1;
        }
        
        for(auto v : simnei) { //i.e., select dominated vertex from u's simnei
            bool tmpf = true;
            for(auto w : CR) if(Matrix[trans[v]][trans[w]] == 1) {
                if(domCR[w] == 0) {
                    tmpf = false;
                    break;
                }
            }
            if(tmpf == true) dom_collection.push_back(v);
        }
        
        if((int)dom_collection.size() > dom_num) {
            ustar = u;
            dom_num = (int) dom_collection.size();
            dom_cand = dom_collection;
        }
        for(auto e : connei) domCR[e] = 0;
    }
    
    for(auto e : dom_cand) domCR[e] = 1;

    vector<ui> tmp_PL = PL;
    assert(PL.size() >= dom_num);
    int idx1 = 0, idx2 = (int) PL.size() - dom_num ;
    for(auto e : tmp_PL) {
        if(domCR[e] == 0) PL[idx1++] = e;
        else PL[idx2++] = e;
    }
    for(auto e : dom_cand) domCR[e] = 0;
#endif
    
    enumT_Dom += emT.elapsed();
    emT.restart();
    
    skipped_dom_ver += dom_cand.size();
    total_ver += PL.size();
    
    for(ui i = 0; i < PL.size() - dom_cand.size(); i++) {
        
        emT.restart();
        
        ui u = PL[i];
        vector<ui> new_CL = CL;
        new_CL.push_back(u);
        
//        cout<<"     *** expansion, will add "<<u<<" into CL."<<endl;

        vector<ui> new_CR;
        for(auto v : CR) if(Matrix[trans[u]][trans[v]] == 1) new_CR.push_back(v);
//        cout<<"     new_CR : "; for(auto e : new_CR) cout<<e<<","; cout<<endl;

        vector<ui> new_PL;
        for(ui j = i + 1; j < PL.size(); j++) {
            ui v = PL[j];
            if(Matrix[trans[u]][trans[v]] == 1) new_PL.push_back(v);
        }
//        cout<<"     new_PL : "; for(auto e : new_PL) cout<<e<<","; cout<<endl;
        
        vector<ui> new_QL;
        for(auto e : QL) if(Matrix[trans[u]][trans[e]] == 1) new_QL.push_back(e);
//        cout<<"     new_QL : "; for(auto e : new_QL) cout<<e<<","; cout<<endl;
        
        enumT_Recur += emT.elapsed();

        Enum_noRSim_adv_core(new_CL, new_CR, new_PL, new_QL);

        QL.push_back(u);

    }
}

void Enum_noRSim_adv_core_noLtau(vector<ui> CL, vector<ui> CR, vector<ui> PL, vector<ui> QL)
{
    if(over_time_flag) return;
    
    double DurTime = (double)clock() / CLOCKS_PER_SEC - each_start_time;
    
    if(DurTime > maxdurtime)
        over_time_flag = true;
//    cout<<"         === in a recursive invoke of Enum_noRSim_adv_core()"<<endl;
//    cout<<"         CL size = "<<CL.size()<<" : "; for(auto e : CL) cout<<e<<" ";cout<<endl;
//    cout<<"         CR size = "<<CR.size()<<" : "; for(auto e : CR) cout<<e<<" ";cout<<endl;
//    cout<<"         PL size = "<<PL.size()<<" : "; for(auto e : PL) cout<<e<<" ";cout<<endl;
//    cout<<"         QL size = "<<QL.size()<<" : "; for(auto e : QL) cout<<e<<" ";cout<<endl;
    
    if(CL.size() + PL.size() < tau || CR.size() < Rtau) {
//        cout<<"         CL.size() + PL.size() < tau || CR.size() < tau, RETURN!!!"<<endl;
        return;
    }
    
    bool maximality = true;
    
    //for each vertex in PL and QL, obtain its degree in CR
//    cout<<"             $$$ for each vertex in PL and QL, obtain its degree in CR"<<endl;
    
    for(auto u : QL) {
        int d = 0;
        for(auto v : CR) {
            if(Matrix[trans[u]][trans[v]] == 1) {
                ++ d;
            }
        }
        if (d == CR.size()) {
            maximality = false;
            /////////////////////early termination
#ifdef _ET_
            bool sim2PL = true;
            for(auto e : PL) {
                if(Matrix[trans[u]][trans[e]] != 1) {
                    sim2PL = false;
                    break;
                }
            }
            if(sim2PL == true) {
                return;
            }
#endif
            /////////////////////early termination
        }
        deg_inCR[u] = d;
    }
    
    for(auto u : PL) {
        int d = 0;
        for(auto v : CR) {
            if(Matrix[trans[u]][trans[v]] == 1) {
                ++ d;
            }
        }
        if (d == CR.size()) {
            maximality = false;
        }
        deg_inCR[u] = d;
    }
    
    if(maximality == true) {
        if(CL.size() >= tau && CR.size() >= Rtau) {
//            cout<<" @@@ find a result : < ";
//            for(auto e : CL) cout<<e<<" "; cout<<"| ";
//            for(auto e : CR) cout<<e<<" "; cout<<">";cout<<endl;

#ifdef _CheckResults_
                vector<ui> C1, C2;
                for(auto e : CL) C1.push_back(oid[e]);
                for(auto e : CR) C2.push_back(oid[e]);
                sort(C1.begin(), C1.end(), less<>());
                sort(C2.begin(), C2.end(), less<>());
                if(C1[0] < C2[0]) results.push_back(make_pair(C1, C2));
                else results.push_back(make_pair(C2, C1));
#endif
                
                ++ MDBC_num;
                if(CL.size() + CR.size() > max_MDBC_size) max_MDBC_size = CL.size() + CR.size();
                if(CL.size() + CR.size() < min_MDBC_size) min_MDBC_size = CL.size() + CR.size();
        }
    }
    
    if(PL.empty()) return;
    
    vector<ui> dom_cand;
    
#ifdef _DOM_
    //domination
    ui ustar;
    int dom_num = -1;
    vector<ui> mixvec = QL;
    for(auto e : PL) mixvec.push_back(e);
    for(auto u : mixvec) {
        //u's similar neighbors in QL and PL
        vector<ui> simnei;
        for(auto v : PL) if(Matrix[trans[u]][trans[v]] == 1) simnei.push_back(v);
        
        //u's connection neighbors in CR
        vector<ui> connei;
        for(auto v : CR) if(Matrix[trans[u]][trans[v]] == 1) {
            connei.push_back(v); domCR[v] = 1;
        }
        vector<ui> dom_collection;
        for(auto v : simnei) {
            vector<ui> neiconnei;
            for(auto w : CR) if(Matrix[trans[v]][trans[w]] == 1) neiconnei.push_back(w);
            bool dom_flag = true;
            for(auto w : neiconnei) if(domCR[w] == 0) {
                dom_flag = false;
                break;
            }
            if(dom_flag == true) dom_collection.push_back(v);
        }
        if((int)dom_collection.size() > dom_num) {
            ustar = u; dom_num = (int) dom_collection.size(); dom_cand = dom_collection;
        }
        for(auto e : connei) domCR[e] = 0;
    }
    assert(dom_num != -1);
    unordered_set<ui> dom_set;
    for(auto e : dom_cand) dom_set.insert(e);
    vector<ui> tmpPL;
    for(auto e : PL) if(dom_set.find(e) == dom_set.end()) tmpPL.push_back(e);
    for(auto e : dom_cand) tmpPL.push_back(e);
    PL = tmpPL;
#endif
    
    for(ui i = 0; i < PL.size() - dom_cand.size(); i++) {
        ui u = PL[i];
        vector<ui> new_CL = CL;
        new_CL.push_back(u);
        
//        cout<<"     *** expansion, will add "<<u<<" into CL."<<endl;

        vector<ui> new_CR;
        for(auto v : CR) if(Matrix[trans[u]][trans[v]] == 1) new_CR.push_back(v);
//        cout<<"     new_CR : "; for(auto e : new_CR) cout<<e<<","; cout<<endl;

        vector<ui> new_PL;
        for(ui j = i + 1; j < PL.size(); j++) {
            ui v = PL[j];
            if(Matrix[trans[u]][trans[v]] == 1) new_PL.push_back(v);
        }
//        cout<<"     new_PL : "; for(auto e : new_PL) cout<<e<<","; cout<<endl;
        
        vector<ui> new_QL;
        for(auto e : QL) if(Matrix[trans[u]][trans[e]] == 1) new_QL.push_back(e);
//        cout<<"     new_QL : "; for(auto e : new_QL) cout<<e<<","; cout<<endl;

        Enum_noRSim_adv_core_noLtau(new_CL, new_CR, new_PL, new_QL);

        QL.push_back(u);

    }
}

void Enum_noRSim_adv_core_Rtau_zero(vector<ui> CL, vector<ui> CR, vector<ui> PL, vector<ui> QL)
{

//    cout<<"         === in a recursive invoke of Enum_noRSim_adv_core()"<<endl;
//    cout<<"         CL size = "<<CL.size()<<" : "; for(auto e : CL) cout<<e<<" ";cout<<endl;
//    cout<<"         CR size = "<<CR.size()<<" : "; for(auto e : CR) cout<<e<<" ";cout<<endl;
//    cout<<"         PL size = "<<PL.size()<<" : "; for(auto e : PL) cout<<e<<" ";cout<<endl;
//    cout<<"         QL size = "<<QL.size()<<" : "; for(auto e : QL) cout<<e<<" ";cout<<endl;
    
    if(CL.size() + PL.size() < tau ) {
//        cout<<"         CL.size() + PL.size() < tau || CR.size() < tau, RETURN!!!"<<endl;
        return;
    }
    
    bool maximality = true;
    
    //for each vertex in PL and QL, obtain its degree in CR
//    cout<<"             $$$ for each vertex in PL and QL, obtain its degree in CR"<<endl;
    
    for(auto u : QL) {
        int d = 0;
        for(auto v : CR) {
            if(Matrix[trans[u]][trans[v]] == 1) {
                ++ d;
            }
        }
        if (d == CR.size()) {
            maximality = false;
            /////////////////////early termination
#ifdef _ET_
            bool sim2PL = true;
            for(auto e : PL) {
                if(Matrix[trans[u]][trans[e]] != 1) {
                    sim2PL = false;
                    break;
                }
            }
            if(sim2PL == true) {
                return;
            }
#endif
            /////////////////////early termination
        }
        deg_inCR[u] = d;
    }
    
    for(auto u : PL) {
        int d = 0;
        for(auto v : CR) {
            if(Matrix[trans[u]][trans[v]] == 1) {
                ++ d;
            }
        }
        if (d == CR.size()) {
            maximality = false;
        }
        deg_inCR[u] = d;
    }
    
    if(maximality == true) {
        if(CL.size() >= tau ) {
//            cout<<" @@@ find a result : < ";
//            for(auto e : CL) cout<<e<<" "; cout<<"| ";
//            for(auto e : CR) cout<<e<<" "; cout<<">";cout<<endl;

#ifdef _CheckResults_
                vector<ui> C1, C2;
                for(auto e : CL) C1.push_back(oid[e]);
                for(auto e : CR) C2.push_back(oid[e]);
                sort(C1.begin(), C1.end(), less<>());
                sort(C2.begin(), C2.end(), less<>());
            if(C1.empty() || C2.empty()) {
                if(C1.empty()) {
                    assert(!C2.empty());
                    results.push_back(make_pair(C2, C1));
                }
                else{
                    assert(!C1.empty());
                    results.push_back(make_pair(C1, C2));
                }
            }
            else{
                if(C1[0] < C2[0]) results.push_back(make_pair(C1, C2));
                else results.push_back(make_pair(C2, C1));
                
            }
#endif
                
                ++ MDBC_num;
                if(CL.size() + CR.size() > max_MDBC_size) max_MDBC_size = CL.size() + CR.size();
                if(CL.size() + CR.size() < min_MDBC_size) min_MDBC_size = CL.size() + CR.size();
        }
    }
    
    if(PL.empty()) return;
    
    vector<ui> dom_cand;
    
#ifdef _DOM_
    //domination
    ui ustar;
    int dom_num = -1;
    vector<ui> mixvec = QL;
    for(auto e : PL) mixvec.push_back(e);
    for(auto u : mixvec) {
        //u's similar neighbors in QL and PL
        vector<ui> simnei;
        for(auto v : PL) if(Matrix[trans[u]][trans[v]] == 1) simnei.push_back(v);
        
        //u's connection neighbors in CR
        vector<ui> connei;
        for(auto v : CR) if(Matrix[trans[u]][trans[v]] == 1) {
            connei.push_back(v); domCR[v] = 1;
        }
        vector<ui> dom_collection;
        for(auto v : simnei) {
            vector<ui> neiconnei;
            for(auto w : CR) if(Matrix[trans[v]][trans[w]] == 1) neiconnei.push_back(w);
            bool dom_flag = true;
            for(auto w : neiconnei) if(domCR[w] == 0) {
                dom_flag = false;
                break;
            }
            if(dom_flag == true) dom_collection.push_back(v);
        }
        if((int)dom_collection.size() > dom_num) {
            ustar = u; dom_num = (int) dom_collection.size(); dom_cand = dom_collection;
        }
        for(auto e : connei) domCR[e] = 0;
    }
    assert(dom_num != -1);
    unordered_set<ui> dom_set;
    for(auto e : dom_cand) dom_set.insert(e);
    vector<ui> tmpPL;
    for(auto e : PL) if(dom_set.find(e) == dom_set.end()) tmpPL.push_back(e);
    for(auto e : dom_cand) tmpPL.push_back(e);
    PL = tmpPL;
#endif
    
    for(ui i = 0; i < PL.size() - dom_cand.size(); i++) {
        ui u = PL[i];
        vector<ui> new_CL = CL;
        new_CL.push_back(u);
        
//        cout<<"     *** expansion, will add "<<u<<" into CL."<<endl;

        vector<ui> new_CR;
        for(auto v : CR) if(Matrix[trans[u]][trans[v]] == 1) new_CR.push_back(v);
//        cout<<"     new_CR : "; for(auto e : new_CR) cout<<e<<","; cout<<endl;

        vector<ui> new_PL;
        for(ui j = i + 1; j < PL.size(); j++) {
            ui v = PL[j];
            if(Matrix[trans[u]][trans[v]] == 1) new_PL.push_back(v);
        }
//        cout<<"     new_PL : "; for(auto e : new_PL) cout<<e<<","; cout<<endl;
        
        vector<ui> new_QL;
        for(auto e : QL) if(Matrix[trans[u]][trans[e]] == 1) new_QL.push_back(e);
//        cout<<"     new_QL : "; for(auto e : new_QL) cout<<e<<","; cout<<endl;

        Enum_noRSim_adv_core_Rtau_zero(new_CL, new_CR, new_PL, new_QL);

        QL.push_back(u);

    }
}

//.
void build_matrix_for_CRPLQL(vector<ui> &CR, vector<ui> &PL, vector<ui> &QL)
{
    for(ui i = 0; i < max_CS_deg; i++) memset(Matrix[i], 0, sizeof(int)*max_CS_deg);
    
    int idx = 0;
    for(auto e : CR) trans[e] = idx ++;
    for(auto e : PL) trans[e] = idx ++;
    for(auto e : QL) trans[e] = idx ++;
    
    //CR
    for(auto u : PL) {
        for(ui i = r_pstart[u]; i < r_pstart[u+1]; i++) {
            ui v = r_edges[i];
            if(inCR[v] == 1) {
                Matrix[trans[u]][trans[v]] = 1;
                Matrix[trans[v]][trans[u]] = 1;
            }
        }
    }
    for(auto u : QL) {
        for(ui i = r_pstart[u]; i < r_pstart[u+1]; i++) {
            ui v = r_edges[i];
            if(inCR[v] == 1) {
                Matrix[trans[u]][trans[v]] = 1;
                Matrix[trans[v]][trans[u]] = 1;
            }
        }
    }
    //L
    for(ui i = 0; i < PL.size(); i++){
        for(ui j = i + 1; j < PL.size(); j++){
            ui u = PL[i], v = PL[j];
            if(js(oid[u], oid[v]) >= epsilon) {
                Matrix[trans[u]][trans[v]] = 1;
                Matrix[trans[v]][trans[u]] = 1;
            }
        }
    }
    for(ui i = 0; i < QL.size(); i++){
        for(ui j = i + 1; j < QL.size(); j++){
            ui u = QL[i], v = QL[j];
            if(js(oid[u], oid[v]) >= epsilon) {
                Matrix[trans[u]][trans[v]] = 1;
                Matrix[trans[v]][trans[u]] = 1;
            }
        }
    }
    for(auto u : QL){
        for(auto v : PL){
            if(js(oid[u], oid[v]) >= epsilon) {
                Matrix[trans[u]][trans[v]] = 1;
                Matrix[trans[v]][trans[u]] = 1;
            }
        }
    }
    
}

void MDBC_Enum_noRSim_adv()
{
    Timer t;
    
    //vertex reduction
    del_ver = new bool[n]; //记录哪些点已被删
    memset(del_ver, 0, sizeof(bool)*n);
    
    if(tau > 1) {
        if(vr_way == 4) index_vr_opt_noRSim();
        else {
            cout<<"no matching vr !"<<endl; exit(1);
        }
    }
    
    cout<<"[1]. VR time cost = "<<integer_to_string(t.elapsed())<<endl;
    t.restart();
    
//    cout<<"remaining vertices : "<<endl;
//    for(ui i = 0; i < n; i ++) {
//        if(del_ver[i] == 0) cout<<i<<endl;
//    }
    
    //rebuild remaining graph
    r_n = 0; r_n1 = 0; r_n2 = 0;
    r_m = 0;
    for(ui u = 0; u < n1; u++) if(del_ver[u] == 0) {
        ++ r_n; ++r_n1;
        r_m += TMPdeg[u];
    }
    for(ui u = n1; u < n; u++) if(del_ver[u] == 0) {
        ++ r_n; ++r_n2;
        r_m += TMPdeg[u];
    }
    assert(r_m%2 == 0); r_m /= 2;
    
    if(r_n == 0) return;
    
    oid = new ui[r_n];
    nid = new ui[n];
    
    ui vididx = 0;
    for(ui u = 0; u < n; u++) if(del_ver[u] == 0) {
//        cout<<u <<" -> "<<vididx<<endl;
        nid[u] = vididx;
        oid[vididx] = u;
        ++ vididx;
    }
    assert(r_n == vididx);
    
    r_pstart = new ui[r_n+1];
    r_edges = new ui[2*r_m];
    r_degree = new int[r_n];
    
    r_pstart[0] = 0;
    ui r_idx = 0;
    for(ui u = 0; u < r_n; u++){
        ui start_pos = r_pstart[r_idx];
        ui tdeg = 0;
        ui orivid = oid[u];
        for(ui i = pstart[orivid]; i < pstart[orivid+1]; i++) if(del_ver[edges[i]] == 0) {
            r_edges[start_pos++] = nid[edges[i]];
            ++ tdeg;
        }
        r_degree[u] = tdeg;
        r_pstart[++r_idx] = start_pos;
    }
    
    assert(r_idx == r_n);
    
//    for(ui i = 0; i < r_n; i++) assert(TMPdeg[oid[i]] == r_degree[i]);
    
//    cout<<"r_pstart info : "<<endl;
//    for(ui i = 0; i < r_n; i++) {
//        cout<<"vertex "<<i<<" neighbors : ";
//        for(ui j = r_pstart[i]; j < r_pstart[i+1]; j++) {
//            cout<<r_edges[j]<<"   ";
//        } cout<<endl;
//    }cout<<endl;
    
    max_CS_deg = 0;
    for(ui i = 0; i < r_n1; i++) if(Sdeg[oid[i]] + r_degree[i] > max_CS_deg)
        max_CS_deg = Sdeg[oid[i]] + r_degree[i];
    
//    cout<<"max_CS_deg = "<<max_CS_deg<<endl;
    
//    for(ui i = 0; i < r_n1; i++) {
//        cout<<"for vertex "<<i<<", its TMPdeg + Sdeg = "<<Sdeg[oid[i]]<<" + "<<r_degree[i]<<" = "<<Sdeg[oid[i]] + r_degree[i]<<endl;
//    }
    
    Matrix = new int * [max_CS_deg];
    for(ui i = 0; i < max_CS_deg; i++) Matrix[i] = new int [max_CS_deg];
    trans = new ui[r_n];
    
    inCR = new int[r_n];
    memset(inCR, 0, sizeof(int)*r_n);
    deg_inCR = new int[r_n];
    domCR = new int[r_n];
    memset(domCR, 0, sizeof(int)*r_n);
    skipped_dom_ver = 0;
    total_ver = 0;
    
    cout<<"[2]. ReBuild time cost = "<<integer_to_string(t.elapsed())<<endl;
    t.restart();
    
    Timer preEnumT;
    
    for(ui u = 0; u < r_n1; u++) {
        
        preEnumT.restart();
        
        each_start_time = (double)clock() / CLOCKS_PER_SEC;
        over_time_flag = false;
        
        //find maximal similar-bicliques containing u
//        if(u%1000==0) {cout<<"@"<<u/1000<<"k,#"<<MDBC_num<<"; "; cout.flush();}
        vector<ui> CL;
        CL.push_back(u);
        
        vector<ui> CR;
        for(ui i = r_pstart[u]; i < r_pstart[u+1]; i++) {
            ui v = r_edges[i];
            CR.push_back(v);
            inCR[v] = 1;
        }
        
        vector<ui> PL, QL;
        
        if(vr_way == 3 || vr_way == 1) {
            vector<ui> vec = get_sim_nei2(oid[u]);
            for(auto & e : vec) {
                ui v = nid[e];
                if(v > u) { PL.push_back(v); }
                else if(v < u) { QL.push_back(v); }
            }
        }
        else if(vr_way == 4) {
            vector<Itval> & cand_list = vsn[oid[u]];
            for(auto & e : cand_list) if(e.max_score >= epsilon) {
                for(ui i = e.s_idx; i <= e.e_idx; i++) if(del_ver[i] == 0) {
                    if(js(oid[u], i) >= epsilon){
                        ui v = nid[i];
                        if(v > u) { PL.push_back(v); }
                        else if(v < u) { QL.push_back(v); }
                    }
                }
            }
        }
        else {cout<<"no matching vr !"<<endl;exit(1);}
        
        preEnum_initSet += preEnumT.elapsed();
        preEnumT.restart();
        
        if((CL.size() + PL.size()) < tau || CR.size() < tau) {
//            cout<<"(CL.size() + PL.size()) < tau || CR.size() < tau, continue !!!"<<endl;
            for(auto e : CR) inCR[e] = 0;
            continue;
        }
        
//        cout<<"CL size = "<<CL.size()<<" : "; for(auto e : CL) cout<<e<<" ";cout<<endl;
//        cout<<"CR size = "<<CR.size()<<" : "; for(auto e : CR) cout<<e<<" ";cout<<endl;
//        cout<<"PL size = "<<PL.size()<<" : "; for(auto e : PL) cout<<e<<" ";cout<<endl;
//        cout<<"QL size = "<<QL.size()<<" : "; for(auto e : QL) cout<<e<<" ";cout<<endl;
        
        preEnumT.restart();
        build_matrix_for_CRPLQL(CR, PL, QL);
        preEnum_buildMatrx += preEnumT.elapsed();
        preEnumT.restart();
        
        Enum_noRSim_adv_core(CL, CR, PL, QL);
        for(auto e : CR) inCR[e] = 0;
        preEnum_Enum += preEnumT.elapsed();
    }
    
    cout<<"[3]. Enum time cost = "<<integer_to_string(t.elapsed())<<endl;
    t.restart();
    
//    long long preenumtotal = preEnum_initSet + preEnum_buildMatrx + preEnum_Enum;
//    cout<<"\t 1 initset = "<<integer_to_string(preEnum_initSet)<<" ("<<(double)preEnum_initSet/preenumtotal * 100<<"%)."<<endl;
//    cout<<"\t 2 buildMa = "<<integer_to_string(preEnum_buildMatrx)<<" ("<<(double)preEnum_buildMatrx/preenumtotal * 100<<"%)."<<endl;
//    cout<<"\t 3 Enum    = "<<integer_to_string(preEnum_Enum)<<" ("<<(double)preEnum_Enum/preenumtotal * 100<<"%)."<<endl;
    
//    long long totalTTT = enumT_Maximal_ET + enumT_Dom + enumT_Sort + enumT_Recur;
//    cout<<"\t\t 1 Max ET : "<<integer_to_string(enumT_Maximal_ET)<<" ("<<(double)enumT_Maximal_ET/totalTTT* 100<<"%)."<<endl;
//    cout<<"\t\t 2 Dom    : "<<integer_to_string(enumT_Dom)<<" ("<<(double)enumT_Dom/totalTTT * 100<<"%)."<<endl;
//    cout<<"\t\t 3 Sort   : "<<integer_to_string(enumT_Sort)<<" ("<<(double)enumT_Sort/totalTTT* 100<<"%)."<<endl;
//    cout<<"\t\t 4 recur  : "<<integer_to_string(enumT_Recur)<<" ("<<(double)enumT_Recur/totalTTT* 100<<"%)."<<endl;
}

void dele_memo()
{
    if(pstart != nullptr){
        delete [] pstart;
        pstart = nullptr;
    }
    if(edges != nullptr){
        delete [] edges;
        edges = nullptr;
    }
    if(degree != nullptr){
        delete [] degree;
        degree = nullptr;
    }
    if(del_ver != nullptr){
        delete [] del_ver;
        del_ver = nullptr;
    }
    if(del_ver_copy != nullptr){
        delete [] del_ver_copy;
        del_ver_copy = nullptr;
    }
    if(edel != nullptr){
        delete [] edel;
        edel = nullptr;
    }
    if(esign != nullptr){
        delete [] esign;
        esign = nullptr;
    }
    if(TMPdeg != nullptr){
        delete [] TMPdeg;
        TMPdeg = nullptr;
    }
    if(Sdeg != nullptr){
        delete [] Sdeg;
        Sdeg = nullptr;
    }
    if(CLp != nullptr){
        delete [] CLp;
        CLp = nullptr;
    }
    if(CLe != nullptr){
        delete [] CLe;
        CLe = nullptr;
    }
    if(r_pstart != nullptr){
        delete [] r_pstart;
        r_pstart = nullptr;
    }
    if(r_edges != nullptr){
        delete [] r_edges;
        r_edges = nullptr;
    }
    if(r_degree != nullptr){
        delete [] r_degree;
        r_degree = nullptr;
    }
    if(peel_s != nullptr){
        delete [] peel_s;
        peel_s = nullptr;
    }
    if(oid != nullptr){
        delete [] oid;
        oid = nullptr;
    }
    if(nid != nullptr){
        delete [] nid;
        nid = nullptr;
    }
    for(int i = 0; i < max_CS_deg; i++) if(Matrix[i] != nullptr){
        delete [] Matrix[i];
        Matrix[i] = nullptr;
    }
    if(trans != nullptr){
        delete [] trans;
        trans = nullptr;
    }
    if(inPQ != nullptr){
        delete [] inPQ;
        inPQ = nullptr;
    }
    if(deg_inP != nullptr){
        delete [] deg_inP;
        deg_inP = nullptr;
    }
    if(pdeg_inP != nullptr){
        delete [] pdeg_inP;
        pdeg_inP = nullptr;
    }
    if(ndeg_inP != nullptr){
        delete [] ndeg_inP;
        ndeg_inP = nullptr;
    }
    if(os != nullptr){
        delete [] os;
        os = nullptr;
    }
    if(inCR != nullptr){
        delete [] inCR;
        inCR = nullptr;
    }
    if(domCR != nullptr){
        delete [] domCR;
        domCR = nullptr;
    }
    if(deg_inCR != nullptr){
        delete [] deg_inCR;
        deg_inCR = nullptr;
    }
    if(comneicntforDBLP != nullptr){
        delete [] comneicntforDBLP;
        comneicntforDBLP = nullptr;
    }
}


void build_index_LG(string graph_name)
{
    cout<<"*** build_index_LG ***"<<endl;
//    cout<<"*** trivial scr: "<<trivial_score<<"  thre_make_seg: "<<thre_make_seg<<"  seg_num_times: "<<seg_num_times<<"  rg_limit: "<<rg_limit<<" ***"<<endl;
    if(rg_limit != 0) {cout<<"in LG, rg_limit must be 0!"<<endl;exit(1);}
    vector<int> INDEX_vid;
    vector<ui> INDEX_flag;
    vector<vector<Itval>> INDEX_list;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    long long total_2hop_nei_size = 0;
    long long total_shrinked_2hop_nei_size = 0;
    
    long long Phi_total = 0;
    long long Phi_exist = 0;
    
    long long make_idvidual = 0;
    long long make_seg = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    for(ui u = 0; u < n; u++){
//        if(u%10000==0) cout<<"v"<<u/10000<<"w "; cout.flush();
//        cout<<" @v"<<u; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        total_2hop_nei_size += two_hop_nei.size();
        
        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            if(simscore >= trivial_score)
                ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }

        total_shrinked_2hop_nei_size += ordered_2hop_neis.size();
        
        ++ Phi_total;
        
        if(ordered_2hop_neis.empty()) continue;
        
        ++ Phi_exist;
        
        sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();
        
        //start to generate summary ranges
        assert(ordered_2hop_neis.size() >= 1);
        
        INDEX_vid.push_back(u);
        
        if(ordered_2hop_neis.size() < thre_make_seg){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(1);
            ++ make_idvidual;
        }
        else{

            vector<Itval> tmpV;
            
            assert(ordered_2hop_neis.size() >= thre_make_seg);
            assert(ordered_2hop_neis.size() >= 2);
            
//            cout<<"ordered nei size = "<<ordered_2hop_neis.size()<<endl;
            int gc = (int)log2(ordered_2hop_neis.size());
            assert(gc < ordered_2hop_neis.size());
//            cout<<"ori gc = "<<gc<<endl;
            gc = gc * seg_num_times;
//            cout<<"     gc * seg_num_times = "<<gc<<endl;
            
            if(gc < 1) gc = 1;
            if(gc > ordered_2hop_neis.size()-1) gc = ordered_2hop_neis.size()-1;
            
//            cout<<"             new gc = "<<gc<<endl<<endl;
            
            // pair < gap value, position in ordered_2hop_neis >
//            priority_queue<pair<int, ui>, vector<pair<int, ui>>, greater<pair<int, ui>>> kset;  //min heap
            priority_queue<pair<int, ui>, vector<pair<int, ui>>, less<pair<int, ui>>> kset;  //max heap //trick
            for(ui i = 0; i < gc; i++){
                int gap_value = ordered_2hop_neis[i+1].first - ordered_2hop_neis[i].first;
                assert(gap_value >= 1);
                kset.push(make_pair(gap_value, i));
            }
            for(ui i = gc; i < ordered_2hop_neis.size() - 1; i++){
                int gap_value = ordered_2hop_neis[i+1].first - ordered_2hop_neis[i].first;
                assert(gap_value >= 1);
                if(gap_value > kset.top().first){
//                    kset.pop();
//                    kset.push(make_pair(gap_value, i));
                }
            }
            assert(kset.size() == gc);
            vector<ui> positions; //store all index in the ordered_2hop_neis.
            while (!kset.empty()) {
                ui idx = kset.top().second;
                positions.push_back(idx);
                kset.pop();
            }
            assert(positions.size() == gc);
            sort(positions.begin(), positions.end(), less<>()); //increasing order
                        
            double minS, maxS;
            ui start_idx = 0;
            for(ui i = 0; i < positions.size(); i++){
                ui end_idx = positions[i];
                assert(end_idx >= start_idx);
                                
                minS = INF;
                maxS = 0;
                for(ui j = start_idx; j <= end_idx; j++){
                    if(ordered_2hop_neis[j].second < minS) minS = ordered_2hop_neis[j].second;
                    if(ordered_2hop_neis[j].second > maxS) maxS = ordered_2hop_neis[j].second;
                }

                tmpV.push_back(Itval(ordered_2hop_neis[start_idx].first, ordered_2hop_neis[end_idx].first, minS, maxS, end_idx-start_idx+1));
                start_idx = end_idx + 1;
            }
            minS = INF;
            maxS = 0;
            for(ui i = start_idx; i < ordered_2hop_neis.size(); i ++){
                if(ordered_2hop_neis[i].second < minS) minS = ordered_2hop_neis[i].second;
                if(ordered_2hop_neis[i].second > maxS) maxS = ordered_2hop_neis[i].second;
            }

            tmpV.push_back(Itval(ordered_2hop_neis[start_idx].first, ordered_2hop_neis[ordered_2hop_neis.size()-1].first, minS, maxS, ordered_2hop_neis.size() - start_idx));
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(2);
            ++ make_seg;
        }

        T_build_ranges += tt.elapsed();
        tt.restart();
        
    }

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
//    cout<<endl<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;

//    cout<<" ### neglect trivial similarity "<<trivial_score<<" (remaining ratio) : "<<(double)total_shrinked_2hop_nei_size/total_2hop_nei_size<<" (i.e., left how many 2-hop neighbors.)"<<endl;
    
    assert(Phi_total == n);
//    cout<<" ### vertex ratio having non-empty Phi : "<<(double)Phi_exist/Phi_total<<" (i.e., need to make seg or make indi.)"<<endl;
    
    assert(make_idvidual + make_seg == Phi_exist);
//    cout<<"     ### make segment ratio : "<<(double)make_seg/Phi_exist<<endl;
    
    assert(INDEX_vid.size() == INDEX_list.size());
    assert(INDEX_vid.size() == INDEX_flag.size());
    
    //binary version
    
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_LG.bin");
    
    FILE * f = Utility::open_file(graph_name.c_str(), "wb");
    
    for(ui i = 0; i < INDEX_vid.size(); i++) {
        fwrite(&INDEX_vid[i], sizeof(int), 1, f);
        int num = INDEX_list[i].size();
        assert(num > 0);
        if(INDEX_flag[i] == 1) {
            num = -num;
            fwrite(&num, sizeof(int), 1, f);
            for(auto &e : INDEX_list[i]) {
                assert(e.e_idx == e.s_idx);
                assert(e.min_score == e.max_score);
                assert(e.c == 1);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                char aaa = (char)(e.min_score*100);
                fwrite(&aaa, sizeof(char), 1, f);
            }
        }
        else {
            fwrite(&num, sizeof(int), 1, f);
            
            for(auto &e : INDEX_list[i]) {
                char a = (char)(e.min_score*100);
                char b = (char)(e.max_score*100);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                fwrite(&e.e_idx, sizeof(ui), 1, f);
                fwrite(&a, sizeof(char), 1, f);
                fwrite(&b, sizeof(char), 1, f);
                fwrite(&e.c, sizeof(int), 1, f);
            }
        }
    }
    fclose(f);
         

    //txt version
     /*
        ofstream fout;
        graph_name.erase(graph_name.end() - 4, graph_name.end());
        graph_name.append("_" + to_string((int)(trivial_score*100)));
        graph_name.append("_" + to_string((int)(thre_make_seg)));
        graph_name.append("_" + to_string((int)(seg_num_times*100)));
        graph_name.append("_" + to_string((int)(rg_limit*100)));
        graph_name.append("_LG.txt");

        //cout<<graph_name<<endl;
        fout.open(graph_name);
        assert(fout.is_open());
        
        for(ui i = 0; i < INDEX_vid.size(); i++){
            if(INDEX_flag[i] == 2 || INDEX_vid[i] == 0) {
                fout<<INDEX_vid[i]<<" ";
                for(auto &e : INDEX_list[i])
                    fout<<e.s_idx<<" "<<e.e_idx<<" "<<(int)(e.min_score*100)<<" "<<(int)(e.max_score*100)<<" "<<e.c<<" ";
                fout<<endl;
            }
            else {
                assert(INDEX_flag[i] == 1);
                assert(INDEX_vid[i] > 0);
                fout<<-INDEX_vid[i]<<" ";
                for(auto &e : INDEX_list[i]){
                    assert(e.e_idx == e.s_idx);
                    assert(e.c == 1);
                    assert(e.min_score == e.max_score);
                    fout<<e.s_idx<<" "<<(int)(e.min_score*100)<<" ";
                }
                fout<<endl;
            }
        }
        fout.close();
    */
    
    delete [] c;
    cout<<"*** finish build_index_LG ***"<<endl;
}

void build_index_LGf(string graph_name)
{
    cout<<"*** build_index_LGf ***"<<endl;
//    cout<<"*** trivial scr: "<<trivial_score<<"  thre_make_seg: "<<thre_make_seg<<"  seg_num_times: "<<seg_num_times<<"  rg_limit: "<<rg_limit<<" ***"<<endl;
    if(rg_limit != 0) {cout<<"in LG, rg_limit must be 0!"<<endl;exit(1);}
    vector<int> INDEX_vid;
    vector<ui> INDEX_flag;
    vector<vector<Itval>> INDEX_list;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    long long total_2hop_nei_size = 0;
    long long total_shrinked_2hop_nei_size = 0;
    
    long long Phi_total = 0;
    long long Phi_exist = 0;
    
    long long make_idvidual = 0;
    long long make_seg = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    for(ui u = 0; u < n; u++){
//        if(u%10000==0) cout<<"v"<<u/10000<<"w "; cout.flush();
//        cout<<" @v"<<u; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        total_2hop_nei_size += two_hop_nei.size();
        
        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            if(simscore >= trivial_score)
                ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }

        total_shrinked_2hop_nei_size += ordered_2hop_neis.size();
        
        ++ Phi_total;
        
        if(ordered_2hop_neis.empty()) continue;
        
        ++ Phi_exist;
        
        sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();
        
        //start to generate summary ranges
        assert(ordered_2hop_neis.size() >= 1);
        
        INDEX_vid.push_back(u);
        
        if(ordered_2hop_neis.size() < thre_make_seg){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(1);
            ++ make_idvidual;
        }
        else{

            vector<Itval> tmpV;
            
            assert(ordered_2hop_neis.size() >= thre_make_seg);
            assert(ordered_2hop_neis.size() >= 2);
            
//            cout<<"ordered nei size = "<<ordered_2hop_neis.size()<<endl;
            int gc = (int)log2(ordered_2hop_neis.size());
            assert(gc < ordered_2hop_neis.size());
//            cout<<"ori gc = "<<gc<<endl;
            gc = gc * seg_num_times;
//            cout<<"     gc * seg_num_times = "<<gc<<endl;
            
            if(gc < 1) gc = 1;
            if(gc > ordered_2hop_neis.size()-1) gc = ordered_2hop_neis.size()-1;
            
//            cout<<"             new gc = "<<gc<<endl<<endl;
            
            // pair < gap value, position in ordered_2hop_neis >
            priority_queue<pair<int, ui>, vector<pair<int, ui>>, greater<pair<int, ui>>> kset;  //min heap
//            priority_queue<pair<int, ui>, vector<pair<int, ui>>, less<pair<int, ui>>> kset;  //max heap //trick
            for(ui i = 0; i < gc; i++){
                int gap_value = ordered_2hop_neis[i+1].first - ordered_2hop_neis[i].first;
                assert(gap_value >= 1);
                kset.push(make_pair(gap_value, i));
            }
            for(ui i = gc; i < ordered_2hop_neis.size() - 1; i++){
                int gap_value = ordered_2hop_neis[i+1].first - ordered_2hop_neis[i].first;
                assert(gap_value >= 1);
                if(gap_value > kset.top().first){
                    kset.pop();
                    kset.push(make_pair(gap_value, i));
                }
            }
            assert(kset.size() == gc);
            vector<ui> positions; //store all index in the ordered_2hop_neis.
            while (!kset.empty()) {
                ui idx = kset.top().second;
                positions.push_back(idx);
                kset.pop();
            }
            assert(positions.size() == gc);
            sort(positions.begin(), positions.end(), less<>()); //increasing order
                        
            double minS, maxS;
            ui start_idx = 0;
            for(ui i = 0; i < positions.size(); i++){
                ui end_idx = positions[i];
                assert(end_idx >= start_idx);
                                
                minS = INF;
                maxS = 0;
                for(ui j = start_idx; j <= end_idx; j++){
                    if(ordered_2hop_neis[j].second < minS) minS = ordered_2hop_neis[j].second;
                    if(ordered_2hop_neis[j].second > maxS) maxS = ordered_2hop_neis[j].second;
                }

                tmpV.push_back(Itval(ordered_2hop_neis[start_idx].first, ordered_2hop_neis[end_idx].first, minS, maxS, end_idx-start_idx+1));
                start_idx = end_idx + 1;
            }
            minS = INF;
            maxS = 0;
            for(ui i = start_idx; i < ordered_2hop_neis.size(); i ++){
                if(ordered_2hop_neis[i].second < minS) minS = ordered_2hop_neis[i].second;
                if(ordered_2hop_neis[i].second > maxS) maxS = ordered_2hop_neis[i].second;
            }

            tmpV.push_back(Itval(ordered_2hop_neis[start_idx].first, ordered_2hop_neis[ordered_2hop_neis.size()-1].first, minS, maxS, ordered_2hop_neis.size() - start_idx));
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(2);
            ++ make_seg;
        }

        T_build_ranges += tt.elapsed();
        tt.restart();
        
    }

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
//    cout<<endl<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;

//    cout<<" ### neglect trivial similarity "<<trivial_score<<" (remaining ratio) : "<<(double)total_shrinked_2hop_nei_size/total_2hop_nei_size<<" (i.e., left how many 2-hop neighbors.)"<<endl;
    
    assert(Phi_total == n);
//    cout<<" ### vertex ratio having non-empty Phi : "<<(double)Phi_exist/Phi_total<<" (i.e., need to make seg or make indi.)"<<endl;
    
    assert(make_idvidual + make_seg == Phi_exist);
//    cout<<"     ### make segment ratio : "<<(double)make_seg/Phi_exist<<endl;
    
    assert(INDEX_vid.size() == INDEX_list.size());
    assert(INDEX_vid.size() == INDEX_flag.size());
    
    //binary version
    
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_LGf.bin");
    
    FILE * f = Utility::open_file(graph_name.c_str(), "wb");
    
    for(ui i = 0; i < INDEX_vid.size(); i++) {
        fwrite(&INDEX_vid[i], sizeof(int), 1, f);
        int num = INDEX_list[i].size();
        assert(num > 0);
        if(INDEX_flag[i] == 1) {
            num = -num;
            fwrite(&num, sizeof(int), 1, f);
            for(auto &e : INDEX_list[i]) {
                assert(e.e_idx == e.s_idx);
                assert(e.min_score == e.max_score);
                assert(e.c == 1);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                char aaa = (char)(e.min_score*100);
                fwrite(&aaa, sizeof(char), 1, f);
            }
        }
        else {
            fwrite(&num, sizeof(int), 1, f);
            
            for(auto &e : INDEX_list[i]) {
                char a = (char)(e.min_score*100);
                char b = (char)(e.max_score*100);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                fwrite(&e.e_idx, sizeof(ui), 1, f);
                fwrite(&a, sizeof(char), 1, f);
                fwrite(&b, sizeof(char), 1, f);
                fwrite(&e.c, sizeof(int), 1, f);
            }
        }
    }
    fclose(f);
         

    //txt version
     /*
        ofstream fout;
        graph_name.erase(graph_name.end() - 4, graph_name.end());
        graph_name.append("_" + to_string((int)(trivial_score*100)));
        graph_name.append("_" + to_string((int)(thre_make_seg)));
        graph_name.append("_" + to_string((int)(seg_num_times*100)));
        graph_name.append("_" + to_string((int)(rg_limit*100)));
        graph_name.append("_LG.txt");

        //cout<<graph_name<<endl;
        fout.open(graph_name);
        assert(fout.is_open());
        
        for(ui i = 0; i < INDEX_vid.size(); i++){
            if(INDEX_flag[i] == 2 || INDEX_vid[i] == 0) {
                fout<<INDEX_vid[i]<<" ";
                for(auto &e : INDEX_list[i])
                    fout<<e.s_idx<<" "<<e.e_idx<<" "<<(int)(e.min_score*100)<<" "<<(int)(e.max_score*100)<<" "<<e.c<<" ";
                fout<<endl;
            }
            else {
                assert(INDEX_flag[i] == 1);
                assert(INDEX_vid[i] > 0);
                fout<<-INDEX_vid[i]<<" ";
                for(auto &e : INDEX_list[i]){
                    assert(e.e_idx == e.s_idx);
                    assert(e.c == 1);
                    assert(e.min_score == e.max_score);
                    fout<<e.s_idx<<" "<<(int)(e.min_score*100)<<" ";
                }
                fout<<endl;
            }
        }
        fout.close();
    */
    
    delete [] c;
    cout<<"*** finish build_index_LGf ***"<<endl;
}

void build_index_rglmt(string graph_name)
{
    cout<<"start to build index rglmt."<<endl;
    
    vector<ui> INDEX_vid;
    vector<vector<Itval>> INDEX_list;
    double RGLMT = 0.1;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    for(ui u = 0; u < n; u++){
        if(u%10000==0) cout<<"process vertex "<<u<<endl;
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
//            cout<<"  its nei "<<v<<endl;
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
//                cout<<"    its 2 hop nei "<<w<<endl;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        if(two_hop_nei.empty()) continue;
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        sort(two_hop_nei.begin(), two_hop_nei.end(), less<>()); //increasing order

        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }
//        for(ui i = 0; i < n; i++) assert(c[i] == 0);
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();
        
        //start to generate summary ranges
        assert(ordered_2hop_neis.size() >= 1);
        INDEX_vid.push_back(u);
        
        vector<Itval> tmpVec;
        
        ui t_size = ordered_2hop_neis.size();
        ui s_idx = 0;
        while (1) {
            pair<ui, double> & s_node = ordered_2hop_neis[s_idx];
            double std_scr = s_node.second;
            Itval tmpI(s_node.first, s_node.first, s_node.second, s_node.second, 1);
            ++ s_idx;
            while (s_idx < t_size) {
                pair<ui, double> & nt_node = ordered_2hop_neis[s_idx];
                assert(nt_node.first > tmpI.e_idx);
                if(fabs(nt_node.second - std_scr) > RGLMT) break;
                tmpI.e_idx = nt_node.first;
                if(nt_node.second < tmpI.min_score) tmpI.min_score = nt_node.second;
                if(nt_node.second > tmpI.max_score) tmpI.max_score = nt_node.second;
                ++ tmpI.c;
                ++ s_idx;
            }
            tmpVec.push_back(tmpI);
            if(s_idx >= t_size) break;
        }
        INDEX_list.push_back(tmpVec);
        
        T_build_ranges += tt.elapsed();
        tt.restart();
    }

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
    cout<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;

    
    assert(INDEX_vid.size() == INDEX_list.size());
    
    ofstream fout;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_rglmt.txt");

    //cout<<graph_name<<endl;
    fout.open(graph_name);
    assert(fout.is_open());
    
    for(ui i = 0; i < INDEX_vid.size(); i++){
        fout<<INDEX_vid[i]<<" ";
        for(auto e : INDEX_list[i]){
            fout<<e.s_idx<<" "<<e.e_idx<<" "<<setprecision(4)<<e.min_score<<" "<<setprecision(4)<<e.max_score<<" "<<e.c<<" ";
        }
        fout<<endl;
    }
    fout.close();
    
    delete [] c;
    cout<<"finish building index rglmt."<<endl;
//    exit(1);
}

void build_index_GR(string graph_name)
{
    cout<<"*** build_index_GR ***"<<endl;
    
    vector<ui> INDEX_vid;
    vector<vector<Itval>> INDEX_list;
    double RGLMT = 0.1;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    
    Timer t1;
    long long t1_allranges = 0, t1_selectrange = 0, t1_partition = 0;
    
    for(ui u = 0; u < n; u++){
        if(u%10000==0) cout<<"v"<<u/10000<<"w "; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
//            cout<<"  its nei "<<v<<endl;
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
//                cout<<"    its 2 hop nei "<<w<<endl;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        if(two_hop_nei.empty()) continue;
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        sort(two_hop_nei.begin(), two_hop_nei.end(), less<>()); //increasing order

        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();

        ui t_size = ordered_2hop_neis.size();
        INDEX_vid.push_back(u);

        if(t_size < 10){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
        }
        else{
            assert(t_size >= 10);
            
            t1.restart();
            
            vector<Range> allRanges;
            for(ui i = 0; i < t_size; i++){
                ui v = ordered_2hop_neis[i].first; //算v的steady range
                double std_scr = ordered_2hop_neis[i].second;
                
                Range tmpR(v, 1, i, i); //Range
                
                if(i > 0){ // need go left
                    ui j = i - 1;
                    assert(j >= 0);
                    while (1) {
                        ui tmpID = ordered_2hop_neis[j].first;
                        double tmpScore = ordered_2hop_neis[j].second;
                        if(fabs(tmpScore - std_scr) > RGLMT) {break;}
                        assert(v > tmpID);
                        tmpR.Lidx = j;
                        if(j == 0) break;
                        -- j;
                    }
                }
                //go right
                ui j = i + 1;
                while (j < t_size) {
                    ui tmpID = ordered_2hop_neis[j].first;
                    double tmpScore = ordered_2hop_neis[j].second;
                    if(fabs(tmpScore - std_scr) > RGLMT) {break;}
                    assert(v < tmpID);
                    tmpR.Ridx = j;
                    ++ j;
                }
                
                assert(tmpR.Ridx >= tmpR.Lidx);
                assert(tmpR.Lidx >= 0 && tmpR.Lidx < t_size);
                assert(tmpR.Ridx >= 0 && tmpR.Ridx < t_size);
                
                tmpR.rgC = ordered_2hop_neis[tmpR.Ridx].first - ordered_2hop_neis[tmpR.Lidx].first + 1;
                allRanges.push_back(tmpR);
            }
            assert(allRanges.size() == t_size);
            
            t1_allranges += t1.elapsed();
            t1.restart();

            int rc = (int)log2(ordered_2hop_neis.size()) - 1;
            assert(rc < t_size);
            
//            vector<Range> rgS;
            vector<pair<ui, ui>> rgS;
            for(ui i = 0; i < rc; i++){ //选出rc个ranges
                int max_rgC = -1;
                ui idx = 0;
                for(ui j = 0; j < allRanges.size(); j++){
                    if(allRanges[j].rgC > max_rgC && c[j] == 0) {
                        max_rgC = allRanges[j].rgC;
                        idx = j;
                    }
                }
                if(max_rgC == -1) break; //说明挑空了
                
                Range selected_range = allRanges[idx];
                rgS.push_back(make_pair(selected_range.Lidx, selected_range.Ridx));
                
                for(ui j = selected_range.Lidx; j <= selected_range.Ridx; j++) c[j] = 1;
                
                //update the influenced ranges
                for(ui j = 0; j < allRanges.size(); j++){
                    if(c[j]) continue;
                    Range & influenced_range = allRanges[j];
                    if(j < selected_range.Lidx) { //在selected range 左侧的ranges
                        assert(influenced_range.Lidx < selected_range.Lidx);
                        assert(influenced_range.Ridx < selected_range.Ridx);
                        if(influenced_range.Ridx >= selected_range.Lidx){
                            assert(selected_range.Lidx >= 1);
                            influenced_range.Ridx = selected_range.Lidx - 1;
                            assert(influenced_range.Lidx <= influenced_range.Ridx);
                            influenced_range.rgC = ordered_2hop_neis[influenced_range.Ridx].first - ordered_2hop_neis[influenced_range.Lidx].first + 1;
                            assert(influenced_range.rgC >= 1);
                        }
                    }
                    else{
                        assert(j > selected_range.Ridx); //在selected range 右侧的ranges
                        assert(influenced_range.Ridx > selected_range.Ridx);
                        assert(influenced_range.Lidx > selected_range.Lidx);
                        if(influenced_range.Lidx <= selected_range.Ridx){
                            influenced_range.Lidx = selected_range.Ridx + 1;
                            assert(influenced_range.Lidx < allRanges.size());
                            assert(influenced_range.Lidx <= influenced_range.Ridx);
                            influenced_range.rgC = ordered_2hop_neis[influenced_range.Ridx].first - ordered_2hop_neis[influenced_range.Lidx].first + 1;
                            assert(influenced_range.rgC >= 1);
                        }
                    }
                }
                
            }
            assert(rgS.size() <= rc);
            for(auto & e : rgS){
                for(ui i = e.first; i <= e.second; i++) c[i] = 0;
            }

            t1_selectrange += t1.elapsed();
            t1.restart();
            
            sort(rgS.begin(), rgS.end(), less<>());
            
            vector<Itval> tmpVec;
            
            ui start_idx = 0;
            for(ui i = 0; i < rgS.size(); i++){
                ui rg_s = rgS[i].first;
                ui rg_e = rgS[i].second;
                assert(rg_e >= rg_s);
                if(rg_s > start_idx){ //存左侧的gap
                    pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                    Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                    ++ start_idx;
                    while (start_idx < rg_s) {
                        pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                        tmpI.e_idx = nxtv.first;
                        if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                        if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                        ++ tmpI.c;
                        ++ start_idx;
                    }
                    assert(start_idx == rg_s);
                    tmpVec.push_back(tmpI);
                }
                //construct rg_s -> rg_e
                pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                for(ui j = rg_s + 1; j <= rg_e; j++){
                    pair<ui, double> & nxtv = ordered_2hop_neis[j];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                }
                tmpVec.push_back(tmpI);
                start_idx = rg_e + 1;
            }
            
            if(start_idx < t_size){
                pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                ++ start_idx;
                while (start_idx < t_size) {
                    pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                    ++ start_idx;
                }
                tmpVec.push_back(tmpI);
            }
            INDEX_list.push_back(tmpVec);
            
            t1_partition += t1.elapsed();
            t1.restart();
            
        } // t_size >= 10
        
        T_build_ranges += tt.elapsed();
        tt.restart();
    } //for each u

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
    cout<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;
    
    
    long long total_t1 = t1_partition + t1_selectrange + t1_allranges;
    
//    cout<<"     t1_allranges = "<<integer_to_string(t1_allranges)<<" ( "<<((double)t1_allranges/(total_t1) )*100<<"% )"<<endl;
//    cout<<"     t1_selectrange = "<<integer_to_string(t1_selectrange)<<" ( "<<((double)t1_selectrange/(total_t1) )*100<<"% )"<<endl;
//    cout<<"     t1_partition = "<<integer_to_string(t1_partition)<<" ( "<<((double)t1_partition/(total_t1) )*100<<"% )"<<endl;

    
    assert(INDEX_vid.size() == INDEX_list.size());
    
    ofstream fout;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GR.txt");

    //cout<<graph_name<<endl;
    fout.open(graph_name);
    assert(fout.is_open());
    
    for(ui i = 0; i < INDEX_vid.size(); i++){
        fout<<INDEX_vid[i]<<" ";
        for(auto e : INDEX_list[i]){
            fout<<e.s_idx<<" "<<e.e_idx<<" "<<setprecision(4)<<e.min_score<<" "<<setprecision(4)<<e.max_score<<" "<<e.c<<" ";
        }
        fout<<endl;
    }
    fout.close();
    
    delete [] c;
    cout<<"*** finish build_index_GR ***"<<endl;

}

void build_index_GRL(string graph_name)
{
    cout<<"*** build_index_GRL ***"<<endl;
    
    vector<ui> INDEX_vid;
    vector<vector<Itval>> INDEX_list;
    double RGLMT = 0.1;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    long long total_2hop_nei_size = 0;
    long long total_shrinked_2hop_nei_size = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    
    Timer t1;
    long long t1_allranges = 0, t1_selectrange = 0, t1_partition = 0;
    
    for(ui u = 0; u < n; u++){
        if(u%10000==0) cout<<"v"<<u/10000<<"w "; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
//            cout<<"  its nei "<<v<<endl;
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
//                cout<<"    its 2 hop nei "<<w<<endl;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        total_2hop_nei_size += two_hop_nei.size();

        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            if(simscore >= 0.05)
                ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }
        
        total_shrinked_2hop_nei_size += ordered_2hop_neis.size();
        
        if(ordered_2hop_neis.empty()) continue;
        sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();

        ui t_size = ordered_2hop_neis.size();
        INDEX_vid.push_back(u);
        
        if(t_size < 8){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
        }
        else{
            assert(t_size >= 8);
            
            t1.restart();
            
            //每一个range里面都是similar score尽可能接近的2hop neis，并且我们想找出跨度最长的几个ranges，vr的时候可能把这些range discard掉
            vector<pair<int, pair<ui, ui>>> allRanges;
            ui s_idx = 0;
            
            while (1) { //linear scan of ordered_2hop_neis, at the same time, record all ranges
                pair<int, pair<ui, ui>> tmpR (1, make_pair(s_idx, s_idx));  //pair< gap_value , pair < start index, end index > >
                double std_scr = ordered_2hop_neis[s_idx].second;
                
                ++ s_idx;
                while (s_idx < t_size) {
                    pair<ui, double> & tnode = ordered_2hop_neis[s_idx];
                    if(fabs(tnode.second - std_scr) > RGLMT) break;
                    assert(tmpR.second.second < s_idx);
                    tmpR.second.second = s_idx;
                    ++ s_idx;
                }
                tmpR.first = ordered_2hop_neis[tmpR.second.second].first - ordered_2hop_neis[tmpR.second.first].first + 1;
                allRanges.push_back(tmpR);
                if(s_idx >= t_size) break;
            }
            
            t1_allranges += t1.elapsed();
            t1.restart();

            int rc = (int)log2(ordered_2hop_neis.size()) - 1;
            assert(rc < t_size);
            
            vector<pair<ui, ui>> rgS;
            if(rc >= allRanges.size()){
                for(auto & e : allRanges)
                    rgS.push_back(e.second);
            }
            else{
                assert(rc < allRanges.size());
                priority_queue<pair<int, pair<ui, ui>>, vector<pair<int, pair<ui, ui>>>, greater<pair<int, pair<ui, ui>>>> kset;
                for(ui i = 0 ; i < rc; i++){
                    pair<int, pair<ui, ui>> & x = allRanges[i];
                    kset.push(x);
                }
                for(ui i = rc; i < allRanges.size(); i++){
                    pair<int, pair<ui, ui>> & x = allRanges[i];
                    if(x.first > kset.top().first){
                        kset.pop();
                        kset.push(x);
                    }
                }
                assert(kset.size() == rc);
                while (!kset.empty()) {
                    rgS.push_back(kset.top().second);
                    kset.pop();
                }
            }
            
            t1_selectrange += t1.elapsed();
            t1.restart();
            
            //rgS现在存的是一个一个的有序的小片段(steady segments)，每个片段的 起始和终止 都对应的是ordered_2hop_neis数组的index
            sort(rgS.begin(), rgS.end(), less<>());
            
            vector<Itval> tmpVec;
            
            ui start_idx = 0;
            for(ui i = 0; i < rgS.size(); i++){
                ui rg_s = rgS[i].first;
                ui rg_e = rgS[i].second;
                assert(rg_e >= rg_s);
                if(start_idx < rg_s){ //存左侧的那一段 2hop neis
                    pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                    Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                    ++ start_idx;
                    while (start_idx < rg_s) {
                        pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                        tmpI.e_idx = nxtv.first;
                        if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                        if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                        ++ tmpI.c;
                        ++ start_idx;
                    }
                    assert(start_idx == rg_s);
                    tmpVec.push_back(tmpI);
                }
                //construct rg_s -> rg_e
                pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                for(ui j = rg_s + 1; j <= rg_e; j++){
                    pair<ui, double> & nxtv = ordered_2hop_neis[j];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                }
                tmpVec.push_back(tmpI);
                start_idx = rg_e + 1;
            }
            
            if(start_idx < t_size){
                pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                ++ start_idx;
                while (start_idx < t_size) {
                    pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                    ++ start_idx;
                }
                tmpVec.push_back(tmpI);
            }
            INDEX_list.push_back(tmpVec);
            
            t1_partition += t1.elapsed();
            t1.restart();
            
        } // t_size >= 4
        
        T_build_ranges += tt.elapsed();
        tt.restart();
    } //for each u

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
    cout<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;
    
//    long long total_t1 = t1_partition + t1_selectrange + t1_allranges;
//    cout<<"     t1_allranges = "<<integer_to_string(t1_allranges)<<" ( "<<((double)t1_allranges/(total_t1) )*100<<"% )"<<endl;
//    cout<<"     t1_selectrange = "<<integer_to_string(t1_selectrange)<<" ( "<<((double)t1_selectrange/(total_t1) )*100<<"% )"<<endl;
//    cout<<"     t1_partition = "<<integer_to_string(t1_partition)<<" ( "<<((double)t1_partition/(total_t1) )*100<<"% )"<<endl;
    
    cout<<"neglect 0.05: "<<total_shrinked_2hop_nei_size<<"/"<<total_2hop_nei_size<<"="<<(double)total_shrinked_2hop_nei_size/total_2hop_nei_size<<endl;

    assert(INDEX_vid.size() == INDEX_list.size());
    
    ofstream fout;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL.txt");

    //cout<<graph_name<<endl;
    fout.open(graph_name);
    assert(fout.is_open());
    
    for(ui i = 0; i < INDEX_vid.size(); i++){
        fout<<INDEX_vid[i]<<" ";
        for(auto e : INDEX_list[i]){
            fout<<e.s_idx<<" "<<e.e_idx<<" "<<setprecision(3)<<e.min_score<<" "<<setprecision(3)<<e.max_score<<" "<<e.c<<" ";
        }
        fout<<endl;
    }
    fout.close();
    
    delete [] c;
    cout<<"*** finish build_index_GRL ***"<<endl;

}

node * build_tree(vector<node *> A)
{
    if(A.size() == 1) {
        return A[0];
    }

    int pos = 0;
    vector<node *> tmp_A;
    while (pos < A.size()) {
        if(pos + 1 < A.size()) {
            node * p1 = A[pos];
            node * p2 = A[pos+1];
            node * new_node = new node;
            new_node->L = p1;
            new_node->R = p2;
            new_node->P = nullptr;
            p1->P = new_node;
            p2->P = new_node;
            new_node->mins = min(p1->mins, p2->mins);
            new_node->maxs = max(p1->maxs, p2->maxs);
            new_node->mark = 0;
            new_node->isv = 0;
            new_node->idx = 0;
            tmp_A.push_back(new_node);
            pos += 2;
        }
        else { //remain 1 node in A
            node * p1 = A[pos];
            node * new_node = new node;
            new_node->L = p1;
            new_node->R = nullptr;
            new_node->P = nullptr;
            p1->P = new_node;
            new_node->mins = p1->mins;
            new_node->maxs = p1->maxs;
            new_node->mark = 0;
            new_node->isv = 0;
            new_node->idx = 0;
            tmp_A.push_back(new_node);
            pos += 2;
        }
    }
//    cout<<" * tmp_A : ";
//    for(auto e : tmp_A) cout<<"["<<e->mins<<","<<e->maxs<<"] "; cout<<endl;
    return build_tree(tmp_A);
}

void show_tree(node * ptr)
{
    if(ptr == nullptr) return;
    if(ptr->isv == 1) {
        cout<<"v"<<ptr->idx<<endl;
    }
    else {
        cout<<"["<<ptr->mins<<","<<ptr->maxs<<"]"<<endl;
    }
    show_tree(ptr->L);
    show_tree(ptr->R);
}

void cp_tree(vector<pair<ui, double>> & ordered_2hop_neis, vector<node *> & A, node * root, vector<pair<ui, int>> & B, double RGLMT)
{
//    cout<<"*** in cp_tree ***"<<endl;
    assert(ordered_2hop_neis.size() == A.size());
    assert(A.size() == B.size());
    assert(root != nullptr);
    
    for(ui i = 0; i < ordered_2hop_neis.size(); i++) {
//        cout<<endl<<" *** processing "<<i<<endl;
        vector<node *> broadway;
        node * unode = A[i];
        
        assert(unode != nullptr);
//        cout<<"unode -> idx = "<<unode->idx<<endl;
        
        while (unode != nullptr) { //find broadway: from unode -> root
            unode->mark = 1;
            broadway.push_back(unode);
            unode = unode->P;
        }
        
//        cout<<"broadway : ";
//        for(auto e : broadway) cout<<"["<<e->mins<<","<<e->maxs<<"]  "; cout<<endl;
        
        assert(unode == nullptr);
        
        double minscore = ordered_2hop_neis[i].second;
        double maxscore = ordered_2hop_neis[i].second;
        
        //broadway is just the path from unode to root, which must exist
        //next, we need to find the downnode pointed by a rightward branch along this broadway, which may be empty
        //downnode is the first node that makes current segment non-steady!
//        cout<<"start to find downnode : "<<endl;
        node * downnode = nullptr;
        for(auto e : broadway) {
            if(e->R != nullptr && e->R->mark == 0) { //e->R may be a downnode
                downnode = e->R;
//                cout<<"     encounter potential downnode : ["<<downnode->mins<<","<<downnode->maxs<<"]."<<endl;
                double tmp_minscore = min(downnode->mins, minscore);
                double tmp_maxscore = max(downnode->maxs, maxscore);
                if((tmp_maxscore - tmp_minscore) > RGLMT) {
//                    cout<<" ### (tmp_maxscore - tmp_minscore) > RGLMT !"<<endl;
                    break; //find the node where we should go down
                }
                minscore = min(downnode->mins, minscore);
                maxscore = max(downnode->maxs, maxscore);
            }
        }
        
        if(downnode == nullptr) { // we are processing the last vertex
//            cout<<"find downnode = nullptr !"<<endl;
            assert(i == (ordered_2hop_neis.size() - 1) );
            B[i].first = i;
            B[i].second = 1;
            
            for (auto e : broadway) e->mark = 0;
            
        }
        else {
            assert(downnode != nullptr);
            
            assert((maxscore - minscore) <= RGLMT);
            
//            cout<<"find downnode = ["<<downnode->mins<<","<<downnode->maxs<<"]."<<endl;
//            cout<<"will go down along this downnode : "<<endl;
            
            while (downnode != nullptr) { //go down till a vertex (leaf)
                if(downnode->isv == 1) { //reach a vertex (leaf)
//                    cout<<"             encounter the leaf v"<<downnode->idx<<endl;
                    ui e_idx = downnode->idx;
                    assert(e_idx > 0);
                    
                    double tmp_minscore = min(downnode->mins, minscore);
                    double tmp_maxscore = max(downnode->maxs, maxscore);
                    
                    if((tmp_maxscore - tmp_minscore) > RGLMT) {
                        -- e_idx;
//                        cout<<"             this node itself is a contradict!"<<endl;
                    }
                    
//                    cout<<"             final e_idx = "<<e_idx<<endl;
                    
                    assert(e_idx >= i);
                    B[i].first = e_idx;
                    B[i].second = ordered_2hop_neis[e_idx].first - ordered_2hop_neis[i].first + 1;
                    for (auto e : broadway) e->mark = 0;
                    break;
                }
                //firstly check its left child
                assert(downnode->L != nullptr);
                double tmp_minscore = min(downnode->L->mins, minscore);
                double tmp_maxscore = max(downnode->L->maxs, maxscore);
                if((tmp_maxscore - tmp_minscore) > RGLMT) {
                    downnode = downnode->L;
//                    cout<<"go to its L branch!"<<endl;
                    continue;
                }
                minscore = min(downnode->L->mins, minscore);
                maxscore = max(downnode->L->maxs, maxscore);
//                cout<<"involve its L branch!"<<endl;
                
                //already involve the left branch
                downnode = downnode->R;
                if(downnode == nullptr) {
//                    cout<<"find an empty R! early terminate!"<<endl;
                    B[i].first = B.size() - 1;
                    B[i].second = ordered_2hop_neis[B.size() - 1].first - ordered_2hop_neis[i].first + 1;
                    for (auto e : broadway) e->mark = 0;
                    break;
                }
            }
        }
    }//for(i)
}

void destroy_tree(node * root)
{
    if(root == nullptr) return;
    destroy_tree(root->L);
    destroy_tree(root->R);
    delete root;
    root = nullptr;
}

void build_index_GRL2(string graph_name)
{
    cout<<"*** build_index_GRL2 ***"<<endl;
//    cout<<"*** trivial scr: "<<trivial_score<<"  thre_make_seg: "<<thre_make_seg<<"  seg_num_times: "<<seg_num_times<<"  rg_limit: "<<rg_limit<<" ***"<<endl;
    
    vector<ui> INDEX_vid;
    vector<ui> INDEX_flag;
    vector<vector<Itval>> INDEX_list;
    double RGLMT = rg_limit;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    long long total_2hop_nei_size = 0;
    long long total_shrinked_2hop_nei_size = 0;
    
    Timer ttt;
    long long T_build_tree = 0;
    long long T_cp_tree = 0;
    long long T_destroy_tree = 0;
    long long T_range_tree = 0;
    
    long long Phi_total = 0;
    long long Phi_exist = 0;
    
    long long make_idvidual = 0;
    long long make_seg = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    
    Timer t1;
    long long t1_allranges = 0, t1_selectrange = 0, t1_partition = 0;
    
    for(ui u = 0; u < n; u++){
        if(u%10000==0) cout<<"v"<<u/10000<<"w "; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        total_2hop_nei_size += two_hop_nei.size();

        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            if(simscore >= trivial_score)
                ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }
        
        total_shrinked_2hop_nei_size += ordered_2hop_neis.size();
        
        ++ Phi_total;
        
        if(ordered_2hop_neis.empty()) continue;
        
        ++ Phi_exist;
        
        sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();

//        ui t_size = ordered_2hop_neis.size();
        
        INDEX_vid.push_back(u);
        
        if(ordered_2hop_neis.size() < thre_make_seg){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(1);
            ++ make_idvidual;
        }
        else{
            
            assert(ordered_2hop_neis.size() >= thre_make_seg);
            assert(ordered_2hop_neis.size() >= 2);
            
            //tree manner
            ttt.restart();
            vector<node *> A;
            vector<pair<ui, int>> B;
            int cnt = 0;
            for(auto &e : ordered_2hop_neis) {
                node * tmp_node = new node;
                tmp_node->L = nullptr;
                tmp_node->R = nullptr;
                tmp_node->P = nullptr;
                tmp_node->mins = e.second;
                tmp_node->maxs = e.second;
                tmp_node->mark = 0;
                tmp_node->isv = 1;
                tmp_node->idx = cnt++;
                A.push_back(tmp_node);
                B.push_back(make_pair(0, 0)); //unexpanded
            }
            assert(cnt == A.size());
            node * tree_root = build_tree(A);
            
            T_build_tree += ttt.elapsed();
            ttt.restart();
            
            //find temporal steady segment for each vertex in ordered_2hop_neis
            cp_tree(ordered_2hop_neis, A, tree_root, B, RGLMT);
            
            T_cp_tree += ttt.elapsed();
            ttt.restart();
                
            destroy_tree(tree_root);
            
            T_destroy_tree += ttt.elapsed();
            ttt.restart();
            //tree manner
            
            //B has been set
            //select rc segments
            int rc = (int)log2(ordered_2hop_neis.size());
            assert(rc < ordered_2hop_neis.size());
            
            rc = rc * seg_num_times;
            
            if(rc > ordered_2hop_neis.size()) rc = ordered_2hop_neis.size();
            
            assert(rc >= 0 && rc <= ordered_2hop_neis.size());
            
            assert(A.size() == B.size());
            assert(A.size() == ordered_2hop_neis.size());
            
//            for(ui i = 0; i < B.size(); i++) assert(B[i].second >= 1);
            
            vector<int> C;
            C.resize(ordered_2hop_neis.size(), 1);
            
            vector<pair<ui, ui>> rgS;
            if(rc == 0) {
                rgS.push_back(make_pair(0, ordered_2hop_neis.size() - 1));
            }
            while (rc > 0) {
                pair<ui, ui> tmp_p;
                int cur_largest_cp = 0;
                for(ui i = 0; i < ordered_2hop_neis.size(); i ++) if(C[i] == 1) {
                    if(B[i].second > cur_largest_cp) {
                        tmp_p = make_pair(i, B[i].first);
                        cur_largest_cp = B[i].second;
                    }
                }
                if(cur_largest_cp == 0) break;
                rgS.push_back(tmp_p);
                assert(tmp_p.first <= tmp_p.second);
                for(ui j = tmp_p.first; j <= tmp_p.second; j++) C[j] = 0;
                
                //update other influenced temporal segments
                for(ui i = 0; i < tmp_p.first; i++) if (C[i] == 1) {
                    if(B[i].first >= tmp_p.first) {
                        assert(tmp_p.first >= 1);
                        ui new_eidx = tmp_p.first - 1;
                        int new_cp = ordered_2hop_neis[new_eidx].first - ordered_2hop_neis[i].first + 1;
                        B[i].first = new_eidx;
                        B[i].second = new_cp;
                    }
                }
                -- rc;
            }
            
//            if(rgS.size() < rc) for(ui i = 0; i < t_size; i++) assert(C[i] == 0);
                
            //rgS现在存的是一个一个的有序的小片段(steady segments)，每个片段的 起始和终止 都对应的是ordered_2hop_neis数组的index
            sort(rgS.begin(), rgS.end(), less<>());
            
            //check rgS
            assert(rgS.size() > 0);
//            if(rgS.size() == 1) {
//                assert(rgS[0].first >= 0 && rgS[0].first < t_size);
//                assert(rgS[0].second >= 0 && rgS[0].second < t_size);
//                assert(rgS[0].first <= rgS[0].second);
//            }
//            else {
//                for(ui i = 0; i < rgS.size() - 1; i++) {
//                    pair<ui, ui> seg1 = rgS[i];
//                    pair<ui, ui> seg2 = rgS[i+1];
//                    assert(seg1.first >= 0 && seg1.first < t_size);
//                    assert(seg1.second >= 0 && seg1.second < t_size);
//                    assert(seg1.first <= seg1.second);
//                    assert(seg2.first >= 0 && seg2.first < t_size);
//                    assert(seg2.second >= 0 && seg2.second < t_size);
//                    assert(seg2.first <= seg2.second);
//                    assert(seg1.second < seg2.first);
//                }
//            }
            
            vector<Itval> tmpVec;
            
            ui start_idx = 0;
            for(ui i = 0; i < rgS.size(); i++){
                ui rg_s = rgS[i].first;
                ui rg_e = rgS[i].second;
                assert(rg_e >= rg_s);
                if(start_idx < rg_s){ //存左侧的那一段 2hop neis
                    pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                    Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                    ++ start_idx;
                    while (start_idx < rg_s) {
                        pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                        tmpI.e_idx = nxtv.first;
                        if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                        if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                        ++ tmpI.c;
                        ++ start_idx;
                    }
                    assert(start_idx == rg_s);
                    tmpVec.push_back(tmpI);
                }
                //construct rg_s -> rg_e
                pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                for(ui j = rg_s + 1; j <= rg_e; j++){
                    pair<ui, double> & nxtv = ordered_2hop_neis[j];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                }
                tmpVec.push_back(tmpI);
                start_idx = rg_e + 1;
            }
            
            if(start_idx < ordered_2hop_neis.size()){
                pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                ++ start_idx;
                while (start_idx < ordered_2hop_neis.size()) {
                    pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                    ++ start_idx;
                }
                tmpVec.push_back(tmpI);
            }
            
            INDEX_list.push_back(tmpVec);
            INDEX_flag.push_back(2);
            ++ make_seg;
            
            T_range_tree += ttt.elapsed();
            ttt.restart();
            
            t1_partition += t1.elapsed();
            t1.restart();
            
        } // t_size >= 4
        
        T_build_ranges += tt.elapsed();
        tt.restart();
    } //for each u

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
//    cout<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;
    
    long long total_tree_T = T_build_tree + T_cp_tree + +T_destroy_tree + T_range_tree;
//    cout<<"         T_build_tree = "<<integer_to_string(T_build_tree)<<" ( "<<((double)T_build_tree/(total_tree_T) )*100<<"% )"<<endl;
//    cout<<"         T_cp_tree = "<<integer_to_string(T_cp_tree)<<" ( "<<((double)T_cp_tree/(total_tree_T) )*100<<"% )"<<endl;
//    cout<<"         T_destroy_tree = "<<integer_to_string(T_destroy_tree)<<" ( "<<((double)T_destroy_tree/(total_tree_T) )*100<<"% )"<<endl;
//    cout<<"         T_range_tree = "<<integer_to_string(T_range_tree)<<" ( "<<((double)T_range_tree/(total_tree_T) )*100<<"% )"<<endl;
    
//    cout<<" ### neglect trivial similarity "<<trivial_score<<" (remaining ratio) : "<<(double)total_shrinked_2hop_nei_size/total_2hop_nei_size<<" (i.e., left how many 2-hop neighbors.)"<<endl;
    
    assert(Phi_total == n);
//    cout<<" ### vertex ratio having non-empty Phi : "<<(double)Phi_exist/Phi_total<<" (i.e., need to make seg or make indi.)"<<endl;
    
    assert(make_idvidual + make_seg == Phi_exist);
//    cout<<"     ### make segment ratio : "<<(double)make_seg/Phi_exist<<endl;
    
    assert(INDEX_vid.size() == INDEX_list.size());
    assert(INDEX_vid.size() == INDEX_flag.size());
    
    //binary version
    
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_GRL2.bin");
    
    FILE * f = Utility::open_file(graph_name.c_str(), "wb");
    
    for(ui i = 0; i < INDEX_vid.size(); i++) {
        fwrite(&INDEX_vid[i], sizeof(int), 1, f);
        int num = INDEX_list[i].size();
        assert(num > 0);
        if(INDEX_flag[i] == 1) {
            num = -num;
            fwrite(&num, sizeof(int), 1, f);
            for(auto &e : INDEX_list[i]) {
                assert(e.e_idx == e.s_idx);
                assert(e.min_score == e.max_score);
                assert(e.c == 1);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                char aaa = (char)(e.min_score*100);
                fwrite(&aaa, sizeof(char), 1, f);
            }
        }
        else {
            fwrite(&num, sizeof(int), 1, f);
            
            for(auto &e : INDEX_list[i]) {
                char a = (char)(e.min_score*100);
                char b = (char)(e.max_score*100);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                fwrite(&e.e_idx, sizeof(ui), 1, f);
                fwrite(&a, sizeof(char), 1, f);
                fwrite(&b, sizeof(char), 1, f);
                fwrite(&e.c, sizeof(int), 1, f);
            }
        }
    }
    fclose(f);
    
    
    //txt version
    /*
    ofstream fout;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL2.txt");

    //cout<<graph_name<<endl;
    fout.open(graph_name);
    assert(fout.is_open());
    
    for(ui i = 0; i < INDEX_vid.size(); i++){
        fout<<INDEX_vid[i]<<" ";
        for(auto e : INDEX_list[i]){
            fout<<e.s_idx<<" "<<e.e_idx<<" "<<setprecision(2)<<e.min_score<<" "<<setprecision(2)<<e.max_score<<" "<<e.c<<" ";
        }
        fout<<endl;
    }
    fout.close();
     */
    
    delete [] c;
    cout<<"*** finish build_index_GRL2 ***"<<endl;

}

void updateQ(deque <pair<double, ui>> & Q, double s, ui j, int flag)
{
    if(flag == 0) {
        while (!Q.empty() && Q.back().first > s) {
            Q.pop_back();
        }
        Q.push_back(make_pair(s, j));
    }
    else {
        while (!Q.empty() && Q.back().first < s) {
            Q.pop_back();
        }
        Q.push_back(make_pair(s, j));
    }
}

void build_index_GRL3(string graph_name)
{
    cout<<"*** build_index_GRL3 ***"<<endl;
    
    vector<ui> INDEX_vid;
    vector<ui> INDEX_flag;
    vector<vector<Itval>> INDEX_list;
    double RGLMT = rg_limit;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    long long total_2hop_nei_size = 0;
    long long total_shrinked_2hop_nei_size = 0;
    
    Timer ttt;
    long long T_build_tree = 0;
    long long T_cp_tree = 0;
    long long T_destroy_tree = 0;
    long long T_range_tree = 0;
    
    long long Phi_total = 0;
    long long Phi_exist = 0;
    
    long long make_idvidual = 0;
    long long make_seg = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    
    Timer t1;
    long long t1_allranges = 0, t1_selectrange = 0, t1_partition = 0;
    
    for(ui u = 0; u < n; u++){
//        if(u%100000==0) cout<<"v"<<u/100000<<"tw "; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        total_2hop_nei_size += two_hop_nei.size();

        vector<pair<ui, double>> ordered_2hop_neis; //pair<vid ,sim>
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            if(simscore >= trivial_score)
                ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }
        
        total_shrinked_2hop_nei_size += ordered_2hop_neis.size();
        
        ++ Phi_total;
        
        if(ordered_2hop_neis.empty()) continue;
        
        ++ Phi_exist;
        
        sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();

//        ui t_size = ordered_2hop_neis.size();
        
        INDEX_vid.push_back(u);
        
        if(ordered_2hop_neis.size() < thre_make_seg){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(1);
            ++ make_idvidual;
        }
        else{
            
            assert(ordered_2hop_neis.size() >= thre_make_seg);
            assert(ordered_2hop_neis.size() >= 2);
            
            //2 queue manner
            ttt.restart();
            vector<pair<ui, int>> B;
            for(ui i = 0; i < ordered_2hop_neis.size(); i++) B.push_back(make_pair(0, 0));
            deque<pair<double, ui>> Q1, Q2; //pair<similarity, pos in ordered_2hop_neis>
            ui i = 0, j = 0;
            Q1.push_back(make_pair(ordered_2hop_neis[i].second, i));
            Q2.push_back(make_pair(ordered_2hop_neis[i].second, i));
            while (i < ordered_2hop_neis.size()) {
                while (Q2.front().first - Q1.front().first <= RGLMT) {
                    ++ j;
                    if(j == ordered_2hop_neis.size()) break;
                    updateQ(Q1, ordered_2hop_neis[j].second, j, 0);
                    updateQ(Q2, ordered_2hop_neis[j].second, j, 1);
                }
                if(j == ordered_2hop_neis.size()) {
                    assert(Q2.front().first - Q1.front().first <= RGLMT);
                    while (i < ordered_2hop_neis.size()) {
                        B[i].first = j - 1;
                        B[i].second = ordered_2hop_neis[j-1].first - ordered_2hop_neis[i].first + 1;
                        ++ i;
                    }
                    break;
                }
                //j reached a position that violates the steady
                assert(j < ordered_2hop_neis.size());
                B[i].first = j - 1;
                B[i].second = ordered_2hop_neis[j-1].first - ordered_2hop_neis[i].first + 1;
                ++ i;
                if(i > Q1.front().second) Q1.pop_front();
                if(i > Q2.front().second) Q2.pop_front();
            }
            //2 queue manner
            
            //B has been set
            //select rc segments
            int rc = (int)log2(ordered_2hop_neis.size());
            assert(rc < ordered_2hop_neis.size());
            
            rc = rc * seg_num_times;
            
            if(rc > ordered_2hop_neis.size()) rc = ordered_2hop_neis.size();
            
            assert(rc >= 0 && rc <= ordered_2hop_neis.size());
            
//            for(ui i = 0; i < B.size(); i++) assert(B[i].second >= 1);
            
            vector<int> C;
            C.resize(ordered_2hop_neis.size(), 1);
            
            vector<pair<ui, ui>> rgS;
            if(rc == 0) {
                rgS.push_back(make_pair(0, ordered_2hop_neis.size() - 1));
            }
            while (rc > 0) {
                pair<ui, ui> tmp_p;
                int cur_largest_cp = 0;
                for(ui i = 0; i < ordered_2hop_neis.size(); i ++) if(C[i] == 1) {
                    if(B[i].second > cur_largest_cp) {
                        tmp_p = make_pair(i, B[i].first);
                        cur_largest_cp = B[i].second;
                    }
                }
                if(cur_largest_cp == 0) break;
                rgS.push_back(tmp_p);
                assert(tmp_p.first <= tmp_p.second);
                for(ui j = tmp_p.first; j <= tmp_p.second; j++) C[j] = 0;
                
                //update other influenced temporal segments
                for(ui i = 0; i < tmp_p.first; i++) if (C[i] == 1) {
                    if(B[i].first >= tmp_p.first) {
                        assert(tmp_p.first >= 1);
                        ui new_eidx = tmp_p.first - 1;
                        int new_cp = ordered_2hop_neis[new_eidx].first - ordered_2hop_neis[i].first + 1;
                        B[i].first = new_eidx;
                        B[i].second = new_cp;
                    }
                }
                -- rc;
            }
            
//            if(rgS.size() < rc) for(ui i = 0; i < t_size; i++) assert(C[i] == 0);
                
            //rgS现在存的是一个一个的有序的小片段(steady segments)，每个片段的 起始和终止 都对应的是ordered_2hop_neis数组的index
            sort(rgS.begin(), rgS.end(), less<>());
            
            //check rgS
            assert(rgS.size() > 0);
//            if(rgS.size() == 1) {
//                assert(rgS[0].first >= 0 && rgS[0].first < t_size);
//                assert(rgS[0].second >= 0 && rgS[0].second < t_size);
//                assert(rgS[0].first <= rgS[0].second);
//            }
//            else {
//                for(ui i = 0; i < rgS.size() - 1; i++) {
//                    pair<ui, ui> seg1 = rgS[i];
//                    pair<ui, ui> seg2 = rgS[i+1];
//                    assert(seg1.first >= 0 && seg1.first < t_size);
//                    assert(seg1.second >= 0 && seg1.second < t_size);
//                    assert(seg1.first <= seg1.second);
//                    assert(seg2.first >= 0 && seg2.first < t_size);
//                    assert(seg2.second >= 0 && seg2.second < t_size);
//                    assert(seg2.first <= seg2.second);
//                    assert(seg1.second < seg2.first);
//                }
//            }
            
            vector<Itval> tmpVec;
            
            ui start_idx = 0;
            for(ui i = 0; i < rgS.size(); i++){
                ui rg_s = rgS[i].first;
                ui rg_e = rgS[i].second;
                assert(rg_e >= rg_s);
                if(start_idx < rg_s){ //存左侧的那一段 2hop neis
                    pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                    Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                    ++ start_idx;
                    while (start_idx < rg_s) {
                        pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                        tmpI.e_idx = nxtv.first;
                        if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                        if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                        ++ tmpI.c;
                        ++ start_idx;
                    }
                    assert(start_idx == rg_s);
                    tmpVec.push_back(tmpI);
                }
                //construct rg_s -> rg_e
                pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                for(ui j = rg_s + 1; j <= rg_e; j++){
                    pair<ui, double> & nxtv = ordered_2hop_neis[j];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                }
                tmpVec.push_back(tmpI);
                start_idx = rg_e + 1;
            }
            
            if(start_idx < ordered_2hop_neis.size()){
                pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                ++ start_idx;
                while (start_idx < ordered_2hop_neis.size()) {
                    pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                    ++ start_idx;
                }
                tmpVec.push_back(tmpI);
            }
            
            INDEX_list.push_back(tmpVec);
            INDEX_flag.push_back(2);
            ++ make_seg;
            
            T_range_tree += ttt.elapsed();
            ttt.restart();
            
            t1_partition += t1.elapsed();
            t1.restart();
            
        } // t_size >= 4
        
        T_build_ranges += tt.elapsed();
        tt.restart();
    } //for each u

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
//    cout<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
//    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;
    
    
//    cout<<" ### neglect trivial similarity "<<trivial_score<<" (remaining ratio) : "<<(double)total_shrinked_2hop_nei_size/total_2hop_nei_size<<" (i.e., left how many 2-hop neighbors.)"<<endl;
    
    assert(Phi_total == n);
//    cout<<" ### vertex ratio having non-empty Phi : "<<(double)Phi_exist/Phi_total<<" (i.e., need to make seg or make indi.)"<<endl;
    
    assert(make_idvidual + make_seg == Phi_exist);
//    cout<<"     ### make segment ratio : "<<(double)make_seg/Phi_exist<<endl;
    
    assert(INDEX_vid.size() == INDEX_list.size());
    assert(INDEX_vid.size() == INDEX_flag.size());
    
    //binary version
    
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_GRL3.bin");
    
    FILE * f = Utility::open_file(graph_name.c_str(), "wb");
    
    for(ui i = 0; i < INDEX_vid.size(); i++) {
        fwrite(&INDEX_vid[i], sizeof(int), 1, f);
        int num = INDEX_list[i].size();
        assert(num > 0);
        if(INDEX_flag[i] == 1) {
            num = -num;
            fwrite(&num, sizeof(int), 1, f);
            for(auto &e : INDEX_list[i]) {
                assert(e.e_idx == e.s_idx);
                assert(e.min_score == e.max_score);
                assert(e.c == 1);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                char aaa = (char)(e.min_score*100);
                fwrite(&aaa, sizeof(char), 1, f);
            }
        }
        else {
            fwrite(&num, sizeof(int), 1, f);
            
            for(auto &e : INDEX_list[i]) {
                char a = (char)(e.min_score*100);
                char b = (char)(e.max_score*100);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                fwrite(&e.e_idx, sizeof(ui), 1, f);
                fwrite(&a, sizeof(char), 1, f);
                fwrite(&b, sizeof(char), 1, f);
                fwrite(&e.c, sizeof(int), 1, f);
            }
        }
    }
    fclose(f);
    
    
    //txt version
    /*
    ofstream fout;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL3.txt");

    //cout<<graph_name<<endl;
    fout.open(graph_name);
    assert(fout.is_open());
    
    for(ui i = 0; i < INDEX_vid.size(); i++){
        fout<<INDEX_vid[i]<<" ";
        for(auto e : INDEX_list[i]){
            fout<<e.s_idx<<" "<<e.e_idx<<" "<<setprecision(2)<<e.min_score<<" "<<setprecision(2)<<e.max_score<<" "<<e.c<<" ";
        }
        fout<<endl;
    }
    fout.close();
     */
    
    delete [] c;
    cout<<"*** finish build_index_GRL3 ***"<<endl;

}

void write_modified_index_to_disk(string graph_name, int flag)
{
//    cout<<"in write_modified_index_to_disk() "<<endl;
    
    if(vsn.empty()) {
        cout<<"vsn is empty."<<endl; exit(1);
    }
    if(flag == 1){
        graph_name.erase(graph_name.end() - 4, graph_name.end());
        graph_name.append("_" + to_string((int)(trivial_score*100)));
        graph_name.append("_" + to_string((int)(thre_make_seg)));
        graph_name.append("_" + to_string((int)(seg_num_times*100)));
        graph_name.append("_" + to_string((int)(rg_limit*100)));
        graph_name.append("_GRL3");
        if(update_type == 1) {
            graph_name.append("_ins");
        }
        else if (update_type == 2) {
            graph_name.append("_del");
        }
        graph_name.append("_" + to_string((num_of_update)));
        graph_name.append(".bin");
//        cout<<"     == > record modified index, name : "<<graph_name<<endl;
    }
    else if (flag == 0) {
        graph_name.erase(graph_name.end() - 4, graph_name.end());
        graph_name.append("_" + to_string((int)(trivial_score*100)));
        graph_name.append("_" + to_string((int)(thre_make_seg)));
        graph_name.append("_" + to_string((int)(seg_num_times*100)));
        graph_name.append("_" + to_string((int)(rg_limit*100)));
        graph_name.append("_GRL3_original.bin");
//        cout<<"     == > prerecord the original index, name : "<<graph_name<<endl;
    }
    else {
        exit(1);
    }
    
    FILE * f = Utility::open_file(graph_name.c_str(), "wb");
    
    for(ui i = 0; i < n; i++) {
        fwrite(&i, sizeof(int), 1, f);
        int num = vsn[i].size();
        if(num <= 0) continue;
        for(auto &e : vsn[i]) {
            char a = (char)(e.min_score*100);
            char b = (char)(e.max_score*100);
            fwrite(&e.s_idx, sizeof(ui), 1, f);
            fwrite(&e.e_idx, sizeof(ui), 1, f);
            fwrite(&a, sizeof(char), 1, f);
            fwrite(&b, sizeof(char), 1, f);
            fwrite(&e.c, sizeof(int), 1, f);
        }
    }
    fclose(f);
    
}

void build_index_GRL4(string graph_name)
{
    cout<<"*** build_index_GRL4 ***"<<endl;
    cout<<"*** trivial scr: "<<trivial_score<<"  thre_make_seg: "<<thre_make_seg<<"  seg_num_times: "<<seg_num_times<<"  rg_limit: "<<rg_limit<<" ***"<<endl;
    
    vector<ui> INDEX_vid;
    vector<ui> INDEX_flag;
    vector<vector<Itval>> INDEX_list;
    double RGLMT = rg_limit;
    
    Timer tt;
    long long T_find_2hopneis = 0;
    long long T_cal_sim_and_sort_for_each = 0;
    long long T_build_ranges = 0;
    long long total_2hop_nei_size = 0;
    long long total_shrinked_2hop_nei_size = 0;
    
    Timer ttt;
    long long T_build_tree = 0;
    long long T_cp_tree = 0;
    long long T_destroy_tree = 0;
    long long T_range_tree = 0;
    
    long long Phi_total = 0;
    long long Phi_exist = 0;
    
    long long make_idvidual = 0;
    long long make_seg = 0;
    
    //for each vertex, we compute its 2-hop neighbors and store them.
    ui * c = new ui[n];
    memset(c, 0, sizeof(ui)*n);
    
    Timer t1;
    long long t1_allranges = 0, t1_selectrange = 0, t1_partition = 0;
    
    for(ui u = 0; u < n; u++){
        if(u%10000==0) cout<<"v"<<u/10000<<"w "; cout.flush();
        tt.restart();
        
        vector<ui> two_hop_nei;
        for(ui i = pstart[u]; i < pstart[u+1]; i++){
            ui v = edges[i];
            for(ui j = pstart[v]; j < pstart[v+1]; j++){
                ui w = edges[j];
                if(w == u) continue;
                if(c[w] == 0) {
                    two_hop_nei.push_back(w);
                }
                ++ c[w];
            }
        }
        
        T_find_2hopneis += tt.elapsed();
        tt.restart();
        
        total_2hop_nei_size += two_hop_nei.size();

        vector<pair<ui, double>> ordered_2hop_neis;
        for(auto e : two_hop_nei){
            assert(degree[u] >= c[e]);
            assert(degree[e] >= c[e]);
            double simscore = (double) c[e] / (degree[u] + degree[e] - c[e]);
            if(simscore >= trivial_score)
                ordered_2hop_neis.push_back(make_pair(e, simscore));
            c[e] = 0;
        }
        
        total_shrinked_2hop_nei_size += ordered_2hop_neis.size();
        
        ++ Phi_total;
        
        if(ordered_2hop_neis.empty()) continue;
        
        ++ Phi_exist;
        
        sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
        
        T_cal_sim_and_sort_for_each += tt.elapsed();
        tt.restart();

//        ui t_size = ordered_2hop_neis.size();
        
        INDEX_vid.push_back(u);
        
        if(ordered_2hop_neis.size() < thre_make_seg){
            vector<Itval> tmpV;
            for(auto e : ordered_2hop_neis){
                tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
            }
            INDEX_list.push_back(tmpV);
            INDEX_flag.push_back(1);
            ++ make_idvidual;
        }
        else{
            
            assert(ordered_2hop_neis.size() >= thre_make_seg);
            assert(ordered_2hop_neis.size() >= 2);
            
            vector<pair<ui, int>> B;
            //?
            
            
            //B has been set
            //select rc segments
            int rc = (int)log2(ordered_2hop_neis.size());
            assert(rc < ordered_2hop_neis.size());
            
            rc = rc * seg_num_times;
            
            if(rc > ordered_2hop_neis.size()) rc = ordered_2hop_neis.size();
            
            assert(rc >= 0 && rc <= ordered_2hop_neis.size());
                                    
            vector<int> C;
            C.resize(ordered_2hop_neis.size(), 1);
            
            vector<pair<ui, ui>> rgS;
            if(rc == 0) {
                rgS.push_back(make_pair(0, ordered_2hop_neis.size() - 1));
            }
            while (rc > 0) {
                pair<ui, ui> tmp_p;
                int cur_largest_cp = 0;
                for(ui i = 0; i < ordered_2hop_neis.size(); i ++) if(C[i] == 1) {
                    if(B[i].second > cur_largest_cp) {
                        tmp_p = make_pair(i, B[i].first);
                        cur_largest_cp = B[i].second;
                    }
                }
                if(cur_largest_cp == 0) break;
                rgS.push_back(tmp_p);
                assert(tmp_p.first <= tmp_p.second);
                for(ui j = tmp_p.first; j <= tmp_p.second; j++) C[j] = 0;
                
                //update other influenced temporal segments
                for(ui i = 0; i < tmp_p.first; i++) if (C[i] == 1) {
                    if(B[i].first >= tmp_p.first) {
                        assert(tmp_p.first >= 1);
                        ui new_eidx = tmp_p.first - 1;
                        int new_cp = ordered_2hop_neis[new_eidx].first - ordered_2hop_neis[i].first + 1;
                        B[i].first = new_eidx;
                        B[i].second = new_cp;
                    }
                }
                -- rc;
            }
            
//            if(rgS.size() < rc) for(ui i = 0; i < t_size; i++) assert(C[i] == 0);
                
            //rgS现在存的是一个一个的有序的小片段(steady segments)，每个片段的 起始和终止 都对应的是ordered_2hop_neis数组的index
            sort(rgS.begin(), rgS.end(), less<>());
            
            //check rgS
            assert(rgS.size() > 0);
//            if(rgS.size() == 1) {
//                assert(rgS[0].first >= 0 && rgS[0].first < t_size);
//                assert(rgS[0].second >= 0 && rgS[0].second < t_size);
//                assert(rgS[0].first <= rgS[0].second);
//            }
//            else {
//                for(ui i = 0; i < rgS.size() - 1; i++) {
//                    pair<ui, ui> seg1 = rgS[i];
//                    pair<ui, ui> seg2 = rgS[i+1];
//                    assert(seg1.first >= 0 && seg1.first < t_size);
//                    assert(seg1.second >= 0 && seg1.second < t_size);
//                    assert(seg1.first <= seg1.second);
//                    assert(seg2.first >= 0 && seg2.first < t_size);
//                    assert(seg2.second >= 0 && seg2.second < t_size);
//                    assert(seg2.first <= seg2.second);
//                    assert(seg1.second < seg2.first);
//                }
//            }
            
            vector<Itval> tmpVec;
            
            ui start_idx = 0;
            for(ui i = 0; i < rgS.size(); i++){
                ui rg_s = rgS[i].first;
                ui rg_e = rgS[i].second;
                assert(rg_e >= rg_s);
                if(start_idx < rg_s){ //存左侧的那一段 2hop neis
                    pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                    Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                    ++ start_idx;
                    while (start_idx < rg_s) {
                        pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                        tmpI.e_idx = nxtv.first;
                        if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                        if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                        ++ tmpI.c;
                        ++ start_idx;
                    }
                    assert(start_idx == rg_s);
                    tmpVec.push_back(tmpI);
                }
                //construct rg_s -> rg_e
                pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                for(ui j = rg_s + 1; j <= rg_e; j++){
                    pair<ui, double> & nxtv = ordered_2hop_neis[j];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                }
                tmpVec.push_back(tmpI);
                start_idx = rg_e + 1;
            }
            
            if(start_idx < ordered_2hop_neis.size()){
                pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                ++ start_idx;
                while (start_idx < ordered_2hop_neis.size()) {
                    pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                    tmpI.e_idx = nxtv.first;
                    if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                    if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                    ++ tmpI.c;
                    ++ start_idx;
                }
                tmpVec.push_back(tmpI);
            }
            
            INDEX_list.push_back(tmpVec);
            INDEX_flag.push_back(2);
            ++ make_seg;
            
            T_range_tree += ttt.elapsed();
            ttt.restart();
            
            t1_partition += t1.elapsed();
            t1.restart();
            
        }
        
        T_build_ranges += tt.elapsed();
        tt.restart();
    } //for each u

    long long totalT = T_find_2hopneis + T_cal_sim_and_sort_for_each + T_build_ranges;
    
    cout<<"     T_find_2hopneis = "<<integer_to_string(T_find_2hopneis)<<" ( "<<((double)T_find_2hopneis/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_cal_sim_and_sort_for_each = "<<integer_to_string(T_cal_sim_and_sort_for_each)<<" ( "<<((double)T_cal_sim_and_sort_for_each/(totalT) )*100<<"% )"<<endl;
    cout<<"     T_build_ranges = "<<integer_to_string(T_build_ranges)<<" ( "<<((double)T_build_ranges/(totalT) )*100<<"% )"<<endl;
    
    long long total_tree_T = T_build_tree + T_cp_tree + +T_destroy_tree + T_range_tree;
    cout<<"         T_build_tree = "<<integer_to_string(T_build_tree)<<" ( "<<((double)T_build_tree/(total_tree_T) )*100<<"% )"<<endl;
    cout<<"         T_cp_tree = "<<integer_to_string(T_cp_tree)<<" ( "<<((double)T_cp_tree/(total_tree_T) )*100<<"% )"<<endl;
    cout<<"         T_destroy_tree = "<<integer_to_string(T_destroy_tree)<<" ( "<<((double)T_destroy_tree/(total_tree_T) )*100<<"% )"<<endl;
    cout<<"         T_range_tree = "<<integer_to_string(T_range_tree)<<" ( "<<((double)T_range_tree/(total_tree_T) )*100<<"% )"<<endl;
    
    cout<<" ### neglect trivial similarity "<<trivial_score<<" (remaining ratio) : "<<(double)total_shrinked_2hop_nei_size/total_2hop_nei_size<<" (i.e., left how many 2-hop neighbors.)"<<endl;
    
    assert(Phi_total == n);
    cout<<" ### vertex ratio having non-empty Phi : "<<(double)Phi_exist/Phi_total<<" (i.e., need to make seg or make indi.)"<<endl;
    
    assert(make_idvidual + make_seg == Phi_exist);
    cout<<"     ### make segment ratio : "<<(double)make_seg/Phi_exist<<endl;
    
    assert(INDEX_vid.size() == INDEX_list.size());
    assert(INDEX_vid.size() == INDEX_flag.size());
    
    //binary version
    
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_" + to_string((int)(trivial_score*100)));
    graph_name.append("_" + to_string((int)(thre_make_seg)));
    graph_name.append("_" + to_string((int)(seg_num_times*100)));
    graph_name.append("_" + to_string((int)(rg_limit*100)));
    graph_name.append("_GRL3.bin");
    
    FILE * f = Utility::open_file(graph_name.c_str(), "wb");
    
    for(ui i = 0; i < INDEX_vid.size(); i++) {
        fwrite(&INDEX_vid[i], sizeof(int), 1, f);
        int num = INDEX_list[i].size();
        assert(num > 0);
        if(INDEX_flag[i] == 1) {
            num = -num;
            fwrite(&num, sizeof(int), 1, f);
            for(auto &e : INDEX_list[i]) {
                assert(e.e_idx == e.s_idx);
                assert(e.min_score == e.max_score);
                assert(e.c == 1);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                char aaa = (char)(e.min_score*100);
                fwrite(&aaa, sizeof(char), 1, f);
            }
        }
        else {
            fwrite(&num, sizeof(int), 1, f);
            
            for(auto &e : INDEX_list[i]) {
                char a = (char)(e.min_score*100);
                char b = (char)(e.max_score*100);
                fwrite(&e.s_idx, sizeof(ui), 1, f);
                fwrite(&e.e_idx, sizeof(ui), 1, f);
                fwrite(&a, sizeof(char), 1, f);
                fwrite(&b, sizeof(char), 1, f);
                fwrite(&e.c, sizeof(int), 1, f);
            }
        }
    }
    fclose(f);
    
    
    //txt version
    /*
    ofstream fout;
    graph_name.erase(graph_name.end() - 4, graph_name.end());
    graph_name.append("_GRL2.txt");

    //cout<<graph_name<<endl;
    fout.open(graph_name);
    assert(fout.is_open());
    
    for(ui i = 0; i < INDEX_vid.size(); i++){
        fout<<INDEX_vid[i]<<" ";
        for(auto e : INDEX_list[i]){
            fout<<e.s_idx<<" "<<e.e_idx<<" "<<setprecision(2)<<e.min_score<<" "<<setprecision(2)<<e.max_score<<" "<<e.c<<" ";
        }
        fout<<endl;
    }
    fout.close();
     */
    
    delete [] c;
    cout<<"*** finish build_index_GRL4 ***"<<endl;

}

void check_results(vector<pair<vector<ui>, vector<ui>>> r)
{
    cout<<"\t = = = = checking results = = = = \t"<<endl;
    if(r.empty()) {
        cout<<"Note that results is empty !!!"<<endl;
    }
    bool flag = true;
    int keytofind = 61;
    vector<pair<vector<ui>, vector<ui>>> tmpr;
    for(auto clique : r) {
        vector<ui> C1 = clique.first;
        vector<ui> C2 = clique.second;
        for(auto e : C1) if (e == keytofind) tmpr.push_back(clique);
        for(auto e : C2) if (e == keytofind) tmpr.push_back(clique);
        //check connection links
        for(auto u : C1) {
            unordered_set<ui> C1nei;
            for(ui i = pstart[u]; i < pstart[u+1]; i++ ) {
                ui v = edges[i];
                C1nei.insert(v);
            }
            for(auto x : C2) {
                if(C1nei.find(x) == C1nei.end()) {
                    cout<<"find a vertex in C2 that is not neighbors of u in C1."<<endl;
                    flag = false;
                    exit(1);
                }
            }
        }

        //check similarity
        for(ui i = 0; i < C1.size(); i++) {
            ui u = C1[i];
            for(ui j = i + 1; j < C1.size(); j++) {
                ui v = C1[j];
                if(js(u, v) < epsilon) {
                    cout<<"find dissimilar pairs in C1."<<endl;
                    flag = false;
                    exit(1);
                }
            }
        }

        if(noRSim == 0) {
            for(ui i = 0; i < C2.size(); i++) {
                ui u = C2[i];
                for(ui j = i + 1; j < C2.size(); j++) {
                    ui v = C2[j];
                    if(js(u, v) < epsilon) {
                        cout<<"find dissimilar pairs in C2."<<endl;
                        flag = false;
                        exit(1);
                    }
                }
            }
        }
    }
    if(flag == false) cout<<"\t = = = = = = WRONG! = = = = = = \t"<<endl;
    else cout<<"\t = = = = = = CORRECT! = = = = = = \t"<<endl;
    
//    for(auto C : tmpr) {
//        cout<<"C : "<<endl;
//        cout<<"\tCL ("<<C.first.size()<<") : "; for(auto e : C.first) cout<<e<<","; cout<<endl;
//        cout<<"\tCR ("<<C.second.size()<<") : "; for(auto e : C.second) cout<<e<<","; cout<<endl;
//    }
//    for(auto C : results) {
//        cout<<"C : "<<endl;
//        cout<<"\tCL ("<<C.first.size()<<") : "; for(auto e : C.first) cout<<e<<","; cout<<endl;
//        cout<<"\tCR ("<<C.second.size()<<") : "; for(auto e : C.second) cout<<e<<","; cout<<endl;
//    }
}

void print_results()
{
    cout<<"Results : "<<endl;
    for(auto e : results) {
        cout<<"\tCL : "; for(auto x : e.first) cout<<x<<", "; cout<<endl;
        cout<<"\tCR : "; for(auto x : e.second) cout<<x<<", "; cout<<endl;
        cout<<"\t----------------------------------------"<<endl;
    }
}


void modify_graph_heu(int flag, vector<pair<ui, ui>> & edges_vect, ui * c, ui * ck)
{
    cout<<"in modify_graph_heu : ";
    
    long long T_get_nei = 0;
    long long T_process_nei = 0;
    Timer tmpt;
    Timer tt;
    
    if(flag == 1){ // insert
        cout<<"INSERT!"<<endl;
        
        for(auto e : edges_vect) {
                        
            ui a = e.first;
            ui b = e.second;
            vector<ui> & anei = G_vv[a]; //G_vv好像是新的图的存储方式，即：vector，便于插删点
            vector<ui>::iterator i;
            for(i = anei.begin(); i < anei.end(); ++i) {
                if(*i > b) break;
            }
            anei.insert(i, b);
            
            vector<ui> & bnei = G_vv[b];
            for(i = bnei.begin(); i < bnei.end(); ++i) {
                if(*i > a) break;
            }
            bnei.insert(i, a);
            
            tmpt.restart();
            /*R-side vertices*/
            vector<ui> twohopnei;
            for(ui a_nei : G_vv[a]) {
                for(ui a_neinei : G_vv[a_nei]) {
                    if(a_neinei == a) continue;
                    if(c[a_neinei] == 0) {
                        twohopnei.push_back(a_neinei);
                    }
                    ++ c[a_neinei];
                }
            }
            
            T_get_nei += tmpt.elapsed();
            
            for(auto x : G_vv[b]) ck[x] = 1;

            tmpt.restart();
            
            for(auto w : twohopnei) {
                ++ total_num_idx;
#ifdef _IdxOpt_
                if(ck[w] == 0) continue;
#endif
                ++ unskipped_num_idx;
                double aw_score = (double)c[w]/(G_vv[a].size() + G_vv[w].size() - c[w]);
                vector<Itval> & tmp_vec = vsn[w];
                vector<Itval>::iterator seg_i;
                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= a && a <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
                        if(c[w] == 1) {
                            (*seg_i).c = (*seg_i).c + 1; //maybe an overestimated c
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec.end()) { //no segment containing a
                    assert(c[w] == 1);
                    char score1 = (char)(aw_score*100);
                    tmp_vec.push_back(Itval(a, a, (double)score1/100, (double)score1/100, 1));
                }
                
                //update \sS_u, search for the segment containing w
                vector<Itval> & tmp_vec_a = vsn[a];
                for(seg_i = tmp_vec_a.begin(); seg_i < tmp_vec_a.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
                        if(c[w] == 1) {
                            (*seg_i).c = (*seg_i).c + 1;
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec_a.end()) { //no segment containing a
                    assert(c[w] == 1);
                    char score1 = (char)(aw_score*100);
                    tmp_vec_a.push_back(Itval(w, w, (double)score1/100, (double)score1/100, 1));
                }
            } //for each w in \Phi_a
            
            T_process_nei += tmpt.elapsed();
            
            for(auto w : twohopnei) c[w] = 0;
            twohopnei.clear();
            
            for(auto x : G_vv[b]) ck[x] = 0;
            
            tmpt.restart();
            /*L-side vertices*/
            for(ui b_nei : G_vv[b]) {
                for(ui b_neinei : G_vv[b_nei]) {
                    if(b_neinei == b) continue;
                    if(c[b_neinei] == 0) {
                        twohopnei.push_back(b_neinei);
                    }
                    ++ c[b_neinei];
                }
            }
            T_get_nei += tmpt.elapsed();
            
            for(auto x : G_vv[a]) ck[x] = 1;
            
            tmpt.restart();
            for(auto w : twohopnei) {
                ++ total_num_idx;
#ifdef _IdxOpt_
                if(ck[w] == 0) continue;
#endif
                ++ unskipped_num_idx;
                double bw_score = (double)c[w]/(G_vv[b].size() + G_vv[w].size() - c[w]);
                //update \sS_w, search for the segment containing b
                vector<Itval> & tmp_vec = vsn[w];
                vector<Itval>::iterator seg_i;
                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= b && b <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
                        if(c[w] == 1) {
                            (*seg_i).c = (*seg_i).c + 1; //maybe an overestimated c
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec.end()) { //no segment containing b
                    assert(c[w] == 1);
                    char score1 = (char)(bw_score*100);
                    tmp_vec.push_back(Itval(b, b, (double)score1/100, (double)score1/100, 1));
                }
                
                //update \sS_u, search for the segment containing w
                vector<Itval> & tmp_vec_b = vsn[b];
                for(seg_i = tmp_vec_b.begin(); seg_i < tmp_vec_b.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
                        if(c[w] == 1) {
                            (*seg_i).c = (*seg_i).c + 1;
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec_b.end()) { //no segment containing b
                    assert(c[w] == 1);
                    char score1 = (char)(bw_score*100);
                    tmp_vec_b.push_back(Itval(w, w, (double)score1/100, (double)score1/100, 1));
                }

            } //for each w in \Phi_a
            
            T_process_nei += tmpt.elapsed();
            
            for(auto w : twohopnei) c[w] = 0;
            
            for(auto x : G_vv[a]) ck[x] = 0;
        
        } //for(auto e : edges_vect)
        
    } //falg = 1
    else if (flag == 2) { // delete
        cout<<"DELETE!"<<endl;
        for(auto e : edges_vect) {
            ui a = e.first;
            ui b = e.second;
            vector<ui> & anei = G_vv[a];
            vector<ui>::iterator i;
            for(i = anei.begin(); i < anei.end(); ++i) {
                if(*i == b) break;
            }
            assert(i < anei.end());
            anei.erase(i);
            
            vector<ui> & bnei = G_vv[b];
            for(i = bnei.begin(); i < bnei.end(); ++i) {
                if(*i == a) break;
            }
            assert(i < bnei.end());
            bnei.erase(i);

            tmpt.restart();
            /*R-side vertices*/
            vector<ui> twohopnei;
            for(ui a_nei : G_vv[a]) {
                for(ui a_neinei : G_vv[a_nei]) {
                    if(a_neinei == a) continue;
                    if(c[a_neinei] == 0) {
                        twohopnei.push_back(a_neinei);
                    }
                    ++ c[a_neinei];
                }
            }
            T_get_nei += tmpt.elapsed();
            
            for(auto x : G_vv[b]) ck[x] = 1;

            tmpt.restart();
            for(auto w : twohopnei) {
                ++ total_num_idx;
#ifdef _IdxOpt_
                if(ck[w] == 1) continue;
#endif
                ++ unskipped_num_idx;
                double aw_score = (double)c[w]/(G_vv[a].size() + G_vv[w].size() - c[w]);
                //update \sS_w, search for the segment containing a
                vector<Itval> & tmp_vec = vsn[w];
                vector<Itval>::iterator seg_i;
                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= a && a <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
                        break;
                    }
                }
                assert(seg_i != tmp_vec.end());
                
                //update \sS_u, search for the segment containing w
                vector<Itval> & tmp_vec_a = vsn[a];
                for(seg_i = tmp_vec_a.begin(); seg_i < tmp_vec_a.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
                        break;
                    }
                }
                assert(seg_i != tmp_vec_a.end());
            } //for each w in \Phi_a
            
            T_process_nei += tmpt.elapsed();
            
            for(auto w : twohopnei) c[w] = 0;
            twohopnei.clear();
            
            for(auto x : G_vv[b]) ck[x] = 0;
            
            tmpt.restart();
            /*L-side vertices*/
            for(ui b_nei : G_vv[b]) {
                for(ui b_neinei : G_vv[b_nei]) {
                    if(b_neinei == b) continue;
                    if(c[b_neinei] == 0) {
                        twohopnei.push_back(b_neinei);
                    }
                    ++ c[b_neinei];
                }
            }
            T_get_nei += tmpt.elapsed();
            
            for(auto x : G_vv[a]) ck[x] = 1;
            
            tmpt.restart();
            for(auto w : twohopnei) {
                ++ total_num_idx;
#ifdef _IdxOpt_
                if(ck[w] == 1) continue;
#endif
                ++ unskipped_num_idx;
                double bw_score = (double)c[w]/(G_vv[b].size() + G_vv[w].size() - c[w]);
                vector<Itval> & tmp_vec = vsn[w];
                vector<Itval>::iterator seg_i;
                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= b && b <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
                        break;
                    }
                }
                assert(seg_i != tmp_vec.end());

                //update \sS_u, search for the segment containing w
                vector<Itval> & tmp_vec_b = vsn[b];
                for(seg_i = tmp_vec_b.begin(); seg_i < tmp_vec_b.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
                        break;
                    }
                }
                assert(seg_i != tmp_vec_b.end());

            } //for each w in \Phi_a
            T_process_nei += tmpt.elapsed();
            
            for(auto w : twohopnei) c[w] = 0;
            
            for(auto x : G_vv[a]) ck[x] = 0;
        
        } //for(auto e : edges_vect)
    }
    else {
        cout<<"nonono!"<<endl;
        exit(1);
    }
    
//    cout<<"\t 1 T_get_nei  = "<<integer_to_string(T_get_nei)<<" ("<<(double)T_get_nei/tt.elapsed() * 100<<"%)."<<endl;
//    cout<<"\t 2 T_proc_nei = "<<integer_to_string(T_process_nei)<<" ("<<(double)T_process_nei/tt.elapsed() * 100<<"%)."<<endl;
    
}

void modify_graph_heu2(int flag, vector<pair<ui, ui>> & edges_vect, ui * c)
{
    cout<<"in modify_graph_heu2 : ";
    
    if(flag == 1){ // insert
        cout<<"INSERT!"<<endl;
        
        for(auto e : edges_vect) {
                        
            ui a = e.first;
            ui b = e.second;
            vector<ui> & anei = G_vv[a];
            vector<ui>::iterator i;
            for(i = anei.begin(); i < anei.end(); ++i) {
                if(*i > b) break;
            }
            anei.insert(i, b);
            
            vector<ui> & bnei = G_vv[b];
            for(i = bnei.begin(); i < bnei.end(); ++i) {
                if(*i > a) break;
            }
            bnei.insert(i, a);
            
            /*R-side vertices*/
            vector<ui> twohopnei;
            for(ui b_nei : G_vv[b]) if(b_nei != a) twohopnei.push_back(b_nei);
            
            for(auto e : G_vv[a]) c[e] = 1;
            for(auto w : twohopnei) {
                ++ total_num_idx;
                ui tmp_cnt = 0;
                for(auto e : G_vv[w]) if(c[e] == 1) ++ tmp_cnt;
//                double aw_score = (double)c[w]/(G_vv[a].size() + G_vv[w].size() - c[w]);
                assert(G_vv[a].size() + G_vv[w].size() >= tmp_cnt);
                double aw_score = (double)tmp_cnt/(G_vv[a].size() + G_vv[w].size() - tmp_cnt);
                vector<Itval> & tmp_vec = vsn[w];
                vector<Itval>::iterator seg_i;
                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= a && a <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
                        if(tmp_cnt == 1) {
                            (*seg_i).c = (*seg_i).c + 1; //maybe an overestimated c
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec.end()) { //no segment containing a
                    assert(tmp_cnt == 1);
                    char score1 = (char)(aw_score*100);
                    tmp_vec.push_back(Itval(a, a, (double)score1/100, (double)score1/100, 1));
                }
                
                //update \sS_u, search for the segment containing w
                vector<Itval> & tmp_vec_a = vsn[a];
                for(seg_i = tmp_vec_a.begin(); seg_i < tmp_vec_a.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
                        if(tmp_cnt == 1) {
                            (*seg_i).c = (*seg_i).c + 1;
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec_a.end()) { //no segment containing a
                    assert(tmp_cnt == 1);
                    char score1 = (char)(aw_score*100);
                    tmp_vec_a.push_back(Itval(w, w, (double)score1/100, (double)score1/100, 1));
                }
            } //for each w in \Phi_a
            for(auto e : G_vv[a]) c[e] = 0;
            
            twohopnei.clear();
                        
            /*L-side vertices*/
            for(auto a_nei : G_vv[a]) if(a_nei != b) twohopnei.push_back(a_nei);
            for(auto e : G_vv[b]) c[e] = 1;
                        
            for(auto w : twohopnei) {
                ++ total_num_idx;
                ui tmp_cnt = 0;
                for(auto e : G_vv[w]) if(c[e] == 1) ++ tmp_cnt;
//                double bw_score = (double)c[w]/(G_vv[b].size() + G_vv[w].size() - c[w]);
                assert(G_vv[b].size() + G_vv[w].size() >= tmp_cnt);
                double bw_score = (double)tmp_cnt/(G_vv[b].size() + G_vv[w].size() - tmp_cnt);
                //update \sS_w, search for the segment containing b
                vector<Itval> & tmp_vec = vsn[w];
                vector<Itval>::iterator seg_i;
                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= b && b <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
                        if(tmp_cnt == 1) {
                            (*seg_i).c = (*seg_i).c + 1; //maybe an overestimated c
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec.end()) { //no segment containing b
                    assert(tmp_cnt == 1);
                    char score1 = (char)(bw_score*100);
                    tmp_vec.push_back(Itval(b, b, (double)score1/100, (double)score1/100, 1));
                }
                
                //update \sS_u, search for the segment containing w
                vector<Itval> & tmp_vec_b = vsn[b];
                for(seg_i = tmp_vec_b.begin(); seg_i < tmp_vec_b.end(); ++ seg_i) {
                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
                        if(tmp_cnt == 1) {
                            (*seg_i).c = (*seg_i).c + 1;
                        }
                        break;
                    }
                }
                if(seg_i == tmp_vec_b.end()) { //no segment containing b
                    assert(tmp_cnt == 1);
                    char score1 = (char)(bw_score*100);
                    tmp_vec_b.push_back(Itval(w, w, (double)score1/100, (double)score1/100, 1));
                }

            } //for each w in \Phi_a
            for(auto e : G_vv[b]) c[e] = 0;
        
        } //for(auto e : edges_vect)
        
    } //falg = 1
//    else if (flag == 2) { // delete
//        cout<<"DELETE!"<<endl;
//        exit(1);
//        for(auto e : edges_vect) {
//            ui a = e.first;
//            ui b = e.second;
//            vector<ui> & anei = G_vv[a];
//            vector<ui>::iterator i;
//            for(i = anei.begin(); i < anei.end(); ++i) {
//                if(*i == b) break;
//            }
//            assert(i < anei.end());
//            anei.erase(i);
//
//            vector<ui> & bnei = G_vv[b];
//            for(i = bnei.begin(); i < bnei.end(); ++i) {
//                if(*i == a) break;
//            }
//            assert(i < bnei.end());
//            bnei.erase(i);
//
//            tmpt.restart();
//            /*R-side vertices*/
//            vector<ui> twohopnei;
//            for(ui a_nei : G_vv[a]) {
//                for(ui a_neinei : G_vv[a_nei]) {
//                    if(a_neinei == a) continue;
//                    if(c[a_neinei] == 0) {
//                        twohopnei.push_back(a_neinei);
//                    }
//                    ++ c[a_neinei];
//                }
//            }
//            T_get_nei += tmpt.elapsed();
//
//            for(auto x : G_vv[b]) ck[x] = 1;
//
//            tmpt.restart();
//            for(auto w : twohopnei) {
//                ++ total_num_idx;
//#ifdef _IdxOpt_
//                if(ck[w] == 1) continue;
//#endif
//                ++ unskipped_num_idx;
//                double aw_score = (double)c[w]/(G_vv[a].size() + G_vv[w].size() - c[w]);
//                //update \sS_w, search for the segment containing a
//                vector<Itval> & tmp_vec = vsn[w];
//                vector<Itval>::iterator seg_i;
//                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
//                    if((*seg_i).s_idx <= a && a <= (*seg_i).e_idx) {
//                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
//                        break;
//                    }
//                }
//                assert(seg_i != tmp_vec.end());
//
//                //update \sS_u, search for the segment containing w
//                vector<Itval> & tmp_vec_a = vsn[a];
//                for(seg_i = tmp_vec_a.begin(); seg_i < tmp_vec_a.end(); ++ seg_i) {
//                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
//                        (*seg_i).max_score = max((*seg_i).max_score, aw_score);
//                        break;
//                    }
//                }
//                assert(seg_i != tmp_vec_a.end());
//            } //for each w in \Phi_a
//
//            T_process_nei += tmpt.elapsed();
//
//            for(auto w : twohopnei) c[w] = 0;
//            twohopnei.clear();
//
//            for(auto x : G_vv[b]) ck[x] = 0;
//
//            tmpt.restart();
//            /*L-side vertices*/
//            for(ui b_nei : G_vv[b]) {
//                for(ui b_neinei : G_vv[b_nei]) {
//                    if(b_neinei == b) continue;
//                    if(c[b_neinei] == 0) {
//                        twohopnei.push_back(b_neinei);
//                    }
//                    ++ c[b_neinei];
//                }
//            }
//            T_get_nei += tmpt.elapsed();
//
//            for(auto x : G_vv[a]) ck[x] = 1;
//
//            tmpt.restart();
//            for(auto w : twohopnei) {
//                ++ total_num_idx;
//#ifdef _IdxOpt_
//                if(ck[w] == 1) continue;
//#endif
//                ++ unskipped_num_idx;
//                double bw_score = (double)c[w]/(G_vv[b].size() + G_vv[w].size() - c[w]);
//                vector<Itval> & tmp_vec = vsn[w];
//                vector<Itval>::iterator seg_i;
//                for(seg_i = tmp_vec.begin(); seg_i < tmp_vec.end(); ++ seg_i) {
//                    if((*seg_i).s_idx <= b && b <= (*seg_i).e_idx) {
//                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
//                        break;
//                    }
//                }
//                assert(seg_i != tmp_vec.end());
//
//                //update \sS_u, search for the segment containing w
//                vector<Itval> & tmp_vec_b = vsn[b];
//                for(seg_i = tmp_vec_b.begin(); seg_i < tmp_vec_b.end(); ++ seg_i) {
//                    if((*seg_i).s_idx <= w && w <= (*seg_i).e_idx) {
//                        (*seg_i).max_score = max((*seg_i).max_score, bw_score);
//                        break;
//                    }
//                }
//                assert(seg_i != tmp_vec_b.end());
//
//            } //for each w in \Phi_a
//            T_process_nei += tmpt.elapsed();
//
//            for(auto w : twohopnei) c[w] = 0;
//
//            for(auto x : G_vv[a]) ck[x] = 0;
//
//        } //for(auto e : edges_vect)
//    }
    else {
        cout<<"nonono!"<<endl;
        exit(1);
    }
}

void modify_graph_bs(int flag, vector<pair<ui, ui>> & edges_vect, ui * c, double RGLMT)
{
    cout<<"in modify_graph_bs"<<endl;
    
//    cout<<"graph : "<<endl;
//    for(ui i = 0; i < n; i++) {
//        cout<<"v "<<i<<": ";
//        vector<ui> & tmp_vec = G_vv[i];
//        for(auto e : tmp_vec) {
//            cout<<e<<", ";
//        }
//        cout<<endl;
//    }
//
//    cout<<"vsn : "<<endl;
//    for(ui i = 0; i < n; i++) {
//        cout<<"v "<<i<<": ";
//        vector<Itval> & tmp_vec = vsn[i];
//        for(auto e : tmp_vec) {
//            cout<<"["<<e.s_idx<<","<<e.e_idx<<"] ("<<e.min_score<<","<<e.max_score<<") "<<e.c<<"   ";
//        }
//        cout<<endl;
//    }
    
    if(flag == 1){ // insert
        for(auto e : edges_vect) {
            ui a = e.first;
            ui b = e.second;
            vector<ui> & anei = G_vv[a];
            vector<ui>::iterator i;
            for(i = anei.begin(); i < anei.end(); ++i) {
                if(*i > b) break;
            }
            anei.insert(i, b);
            
            vector<ui> & bnei = G_vv[b];
            for(i = bnei.begin(); i < bnei.end(); ++i) {
                if(*i > a) break;
            }
            bnei.insert(i, a);
            
            
            /*R-side vertices*/
            vector<ui> todo_list;
            for(ui a_nei : G_vv[a]) {
//                cout<<"a_nei = "<<a_nei<<endl;
                for(ui a_neinei : G_vv[a_nei]) {
                    if(a_neinei == a) continue;
                    if(c[a_neinei] == 0) {
//                        cout<<"     pushback "<<a_neinei<<endl;
                        todo_list.push_back(a_neinei);
                    }
                    ++ c[a_neinei];
                }
            }
            for (auto x : todo_list) c[x] = 0;
            
            todo_list.push_back(a);
            
            for (auto element : todo_list) {
                
                vector<ui> two_hop_nei;
                for(ui a_nei : G_vv[element]) {
                    for(ui a_neinei : G_vv[a_nei]) {
                        if(a_neinei == element) continue;
                        if(c[a_neinei] == 0) {
                            two_hop_nei.push_back(a_neinei);
                        }
                        ++ c[a_neinei];
                    }
                }
                vector<pair<ui, double>> ordered_2hop_neis;
                for(auto each : two_hop_nei){
                    double simscore = (double) c[each] / (G_vv[each].size() + G_vv[element].size() - c[each]);
                    if(simscore >= trivial_score)
                        ordered_2hop_neis.push_back(make_pair(each, simscore));
                    c[each] = 0;
                }
                if(ordered_2hop_neis.empty()) continue;
                sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
                if(ordered_2hop_neis.size() < thre_make_seg){
                    vector<Itval> tmpV;
                    for(auto e : ordered_2hop_neis){
                        tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
                    }
                    vsn[element] = tmpV;
                }
                else {
                    
                    //tree manner
                    vector<node *> A;
                    vector<pair<ui, int>> B;
                    int cnt = 0;
                    for(auto &x : ordered_2hop_neis) {
                        node * tmp_node = new node;
                        tmp_node->L = nullptr;
                        tmp_node->R = nullptr;
                        tmp_node->P = nullptr;
                        tmp_node->mins = x.second;
                        tmp_node->maxs = x.second;
                        tmp_node->mark = 0;
                        tmp_node->isv = 1;
                        tmp_node->idx = cnt++;
                        A.push_back(tmp_node);
                        B.push_back(make_pair(0, 0)); //unexpanded
                    }
                    assert(cnt == A.size());
                    node * tree_root = build_tree(A);
                    
                    
                    //find temporal steady segment for each vertex in ordered_2hop_neis
                    cp_tree(ordered_2hop_neis, A, tree_root, B, RGLMT);
                    

                    destroy_tree(tree_root);
                    
                    //tree manner
                    
                    //B has been set
                    //select rc segments
                    int rc = (int)log2(ordered_2hop_neis.size());
                    assert(rc < ordered_2hop_neis.size());
                    
                    rc = rc * seg_num_times;
                    
                    if(rc > ordered_2hop_neis.size()) rc = ordered_2hop_neis.size();
                    
                    assert(rc >= 0 && rc <= ordered_2hop_neis.size());
                    
                    assert(A.size() == B.size());
                    assert(A.size() == ordered_2hop_neis.size());
                    
        //            for(ui i = 0; i < B.size(); i++) assert(B[i].second >= 1);
                    
                    vector<int> C;
                    C.resize(ordered_2hop_neis.size(), 1);
                    
                    vector<pair<ui, ui>> rgS;
                    if(rc == 0) {
                        rgS.push_back(make_pair(0, ordered_2hop_neis.size() - 1));
                    }
                    while (rc > 0) {
                        pair<ui, ui> tmp_p;
                        int cur_largest_cp = 0;
                        for(ui i = 0; i < ordered_2hop_neis.size(); i ++) if(C[i] == 1) {
                            if(B[i].second > cur_largest_cp) {
                                tmp_p = make_pair(i, B[i].first);
                                cur_largest_cp = B[i].second;
                            }
                        }
                        if(cur_largest_cp == 0) break;
                        rgS.push_back(tmp_p);
                        assert(tmp_p.first <= tmp_p.second);
                        for(ui j = tmp_p.first; j <= tmp_p.second; j++) C[j] = 0;
                        
                        //update other influenced temporal segments
                        for(ui i = 0; i < tmp_p.first; i++) if (C[i] == 1) {
                            if(B[i].first >= tmp_p.first) {
                                assert(tmp_p.first >= 1);
                                ui new_eidx = tmp_p.first - 1;
                                int new_cp = ordered_2hop_neis[new_eidx].first - ordered_2hop_neis[i].first + 1;
                                B[i].first = new_eidx;
                                B[i].second = new_cp;
                            }
                        }
                        -- rc;
                    }
                    
        //            if(rgS.size() < rc) for(ui i = 0; i < t_size; i++) assert(C[i] == 0);
                        
                    //rgS现在存的是一个一个的有序的小片段(steady segments)，每个片段的 起始和终止 都对应的是ordered_2hop_neis数组的index
                    sort(rgS.begin(), rgS.end(), less<>());
                    
                    //check rgS
                    assert(rgS.size() > 0);
                    
                    vector<Itval> tmpVec;
                    
                    ui start_idx = 0;
                    for(ui i = 0; i < rgS.size(); i++){
                        ui rg_s = rgS[i].first;
                        ui rg_e = rgS[i].second;
                        assert(rg_e >= rg_s);
                        if(start_idx < rg_s){ //存左侧的那一段 2hop neis
                            pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                            Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                            ++ start_idx;
                            while (start_idx < rg_s) {
                                pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                                tmpI.e_idx = nxtv.first;
                                if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                                if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                                ++ tmpI.c;
                                ++ start_idx;
                            }
                            assert(start_idx == rg_s);
                            tmpVec.push_back(tmpI);
                        }
                        //construct rg_s -> rg_e
                        pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                        Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                        for(ui j = rg_s + 1; j <= rg_e; j++){
                            pair<ui, double> & nxtv = ordered_2hop_neis[j];
                            tmpI.e_idx = nxtv.first;
                            if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                            if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                            ++ tmpI.c;
                        }
                        tmpVec.push_back(tmpI);
                        start_idx = rg_e + 1;
                    }
                    
                    if(start_idx < ordered_2hop_neis.size()){
                        pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                        Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                        ++ start_idx;
                        while (start_idx < ordered_2hop_neis.size()) {
                            pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                            tmpI.e_idx = nxtv.first;
                            if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                            if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                            ++ tmpI.c;
                            ++ start_idx;
                        }
                        tmpVec.push_back(tmpI);
                    }
                    
                    vsn[element] = tmpVec;
                    
                }
            }
            
            /*L-side vertices*/
            todo_list.clear();
            for(ui a_nei : G_vv[b]) {
//                cout<<"a_nei = "<<a_nei<<endl;
                for(ui a_neinei : G_vv[a_nei]) {
                    if(a_neinei == b) continue;
                    if(c[a_neinei] == 0) {
//                        cout<<"     pushback "<<a_neinei<<endl;
                        todo_list.push_back(a_neinei);
                    }
                    ++ c[a_neinei];
                }
            }
            for (auto x : todo_list) c[x] = 0;
            
            todo_list.push_back(b);
            
            for (auto element : todo_list) {
                
                vector<ui> two_hop_nei;
                for(ui a_nei : G_vv[element]) {
                    for(ui a_neinei : G_vv[a_nei]) {
                        if(a_neinei == element) continue;
                        if(c[a_neinei] == 0) {
                            two_hop_nei.push_back(a_neinei);
                        }
                        ++ c[a_neinei];
                    }
                }
                vector<pair<ui, double>> ordered_2hop_neis;
                for(auto each : two_hop_nei){
                    double simscore = (double) c[each] / (G_vv[each].size() + G_vv[element].size() - c[each]);
                    if(simscore >= trivial_score)
                        ordered_2hop_neis.push_back(make_pair(each, simscore));
                    c[each] = 0;
                }
                if(ordered_2hop_neis.empty()) continue;
                sort(ordered_2hop_neis.begin(), ordered_2hop_neis.end(), less<>()); //increasing order
                if(ordered_2hop_neis.size() < thre_make_seg){
                    vector<Itval> tmpV;
                    for(auto e : ordered_2hop_neis){
                        tmpV.push_back(Itval(e.first, e.first, e.second, e.second, 1));
                    }
                    vsn[element] = tmpV;
                }
                else {
                    
                    //tree manner
                    vector<node *> A;
                    vector<pair<ui, int>> B;
                    int cnt = 0;
                    for(auto &x : ordered_2hop_neis) {
                        node * tmp_node = new node;
                        tmp_node->L = nullptr;
                        tmp_node->R = nullptr;
                        tmp_node->P = nullptr;
                        tmp_node->mins = x.second;
                        tmp_node->maxs = x.second;
                        tmp_node->mark = 0;
                        tmp_node->isv = 1;
                        tmp_node->idx = cnt++;
                        A.push_back(tmp_node);
                        B.push_back(make_pair(0, 0)); //unexpanded
                    }
                    assert(cnt == A.size());
                    node * tree_root = build_tree(A);
                    
                    
                    //find temporal steady segment for each vertex in ordered_2hop_neis
                    cp_tree(ordered_2hop_neis, A, tree_root, B, RGLMT);
                    

                    destroy_tree(tree_root);
                    
                    //tree manner
                    
                    //B has been set
                    //select rc segments
                    int rc = (int)log2(ordered_2hop_neis.size());
                    assert(rc < ordered_2hop_neis.size());
                    
                    rc = rc * seg_num_times;
                    
                    if(rc > ordered_2hop_neis.size()) rc = ordered_2hop_neis.size();
                    
                    assert(rc >= 0 && rc <= ordered_2hop_neis.size());
                    
                    assert(A.size() == B.size());
                    assert(A.size() == ordered_2hop_neis.size());
                    
        //            for(ui i = 0; i < B.size(); i++) assert(B[i].second >= 1);
                    
                    vector<int> C;
                    C.resize(ordered_2hop_neis.size(), 1);
                    
                    vector<pair<ui, ui>> rgS;
                    if(rc == 0) {
                        rgS.push_back(make_pair(0, ordered_2hop_neis.size() - 1));
                    }
                    while (rc > 0) {
                        pair<ui, ui> tmp_p;
                        int cur_largest_cp = 0;
                        for(ui i = 0; i < ordered_2hop_neis.size(); i ++) if(C[i] == 1) {
                            if(B[i].second > cur_largest_cp) {
                                tmp_p = make_pair(i, B[i].first);
                                cur_largest_cp = B[i].second;
                            }
                        }
                        if(cur_largest_cp == 0) break;
                        rgS.push_back(tmp_p);
                        assert(tmp_p.first <= tmp_p.second);
                        for(ui j = tmp_p.first; j <= tmp_p.second; j++) C[j] = 0;
                        
                        //update other influenced temporal segments
                        for(ui i = 0; i < tmp_p.first; i++) if (C[i] == 1) {
                            if(B[i].first >= tmp_p.first) {
                                assert(tmp_p.first >= 1);
                                ui new_eidx = tmp_p.first - 1;
                                int new_cp = ordered_2hop_neis[new_eidx].first - ordered_2hop_neis[i].first + 1;
                                B[i].first = new_eidx;
                                B[i].second = new_cp;
                            }
                        }
                        -- rc;
                    }
                    
        //            if(rgS.size() < rc) for(ui i = 0; i < t_size; i++) assert(C[i] == 0);
                        
                    //rgS现在存的是一个一个的有序的小片段(steady segments)，每个片段的 起始和终止 都对应的是ordered_2hop_neis数组的index
                    sort(rgS.begin(), rgS.end(), less<>());
                    
                    //check rgS
                    assert(rgS.size() > 0);
                    
                    vector<Itval> tmpVec;
                    
                    ui start_idx = 0;
                    for(ui i = 0; i < rgS.size(); i++){
                        ui rg_s = rgS[i].first;
                        ui rg_e = rgS[i].second;
                        assert(rg_e >= rg_s);
                        if(start_idx < rg_s){ //存左侧的那一段 2hop neis
                            pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                            Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                            ++ start_idx;
                            while (start_idx < rg_s) {
                                pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                                tmpI.e_idx = nxtv.first;
                                if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                                if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                                ++ tmpI.c;
                                ++ start_idx;
                            }
                            assert(start_idx == rg_s);
                            tmpVec.push_back(tmpI);
                        }
                        //construct rg_s -> rg_e
                        pair<ui, double> & tmpv = ordered_2hop_neis[rg_s];
                        Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                        for(ui j = rg_s + 1; j <= rg_e; j++){
                            pair<ui, double> & nxtv = ordered_2hop_neis[j];
                            tmpI.e_idx = nxtv.first;
                            if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                            if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                            ++ tmpI.c;
                        }
                        tmpVec.push_back(tmpI);
                        start_idx = rg_e + 1;
                    }
                    
                    if(start_idx < ordered_2hop_neis.size()){
                        pair<ui, double> & tmpv = ordered_2hop_neis[start_idx];
                        Itval tmpI(tmpv.first, tmpv.first, tmpv.second, tmpv.second, 1);
                        ++ start_idx;
                        while (start_idx < ordered_2hop_neis.size()) {
                            pair<ui, double> & nxtv = ordered_2hop_neis[start_idx];
                            tmpI.e_idx = nxtv.first;
                            if(nxtv.second < tmpI.min_score) tmpI.min_score = nxtv.second;
                            if(nxtv.second > tmpI.max_score) tmpI.max_score = nxtv.second;
                            ++ tmpI.c;
                            ++ start_idx;
                        }
                        tmpVec.push_back(tmpI);
                    }
                    
                    vsn[element] = tmpVec;
                    
                }
            }
        
        } //for(auto e : edges_vect)
        
    } //falg = 1
    else if (flag == 2) { // delete
        //...
    }
    else {
        exit(1);
    }
}

int main(int argc, const char * argv[]) {
    string graph_name = argv[1];
    load_graph_binary(graph_name);
    
    cout<<"*** ";
#ifdef _PreORDER_
    cout<<" _PreORDER_ ,";
#else
    cout<<" NO _PreORDER_ ,";
#endif
#ifdef _ORDER_
    cout<<" _ORDER_ ,";
#else
    cout<<" NO _ORDER_ ,";
#endif
#ifdef _DOM_
    cout<<" _DOM_ ,";
#else
    cout<<" NO _DOM_ ,";
#endif
#ifdef _ET_
    cout<<" _ET_ ,";
#else
    cout<<" NO _ET_ ,";
#endif
    
#ifdef _SHRINK_
    cout<<" _SHRINK_ ,";
#else
    cout<<" NO _SHRINK_ ,";
#endif
#ifdef _JSUB_
    cout<<" _JSUB_ ,";
#else
    cout<<" NO _JSUB_ ,";
#endif
#ifdef _IdxOpt_
    cout<<" _IdxOpt_ ";
#else
    cout<<" NO _IdxOpt_ ";
#endif
    cout<<" ***"<<endl;

    seg_num_times = atof(argv[3]);

    rg_limit = atof(argv[4]);
    assert(rg_limit >= 0 && rg_limit <= 1);

    int bld_idx = atoi(argv[2]);
    if(bld_idx == 1){
        assert(argc >= 6);
        Timer t;

        string way = argv[5];
        if (way.compare("LG") == 0) build_index_LG(graph_name);
        else if (way.compare("LGf") == 0) build_index_LGf(graph_name);
        else if (way.compare("GRL2") == 0) build_index_GRL2(graph_name); // segment tree
        else if (way.compare("GRL3") == 0) build_index_GRL3(graph_name); // TPA
        else cout<<"no matching build index way."<<endl;

        cout<<"Building index time cost = "<<integer_to_string(t.elapsed())<<endl;
        dele_memo();
        return 0;
    }

    int lod_idx = atoi(argv[6]);
    if(lod_idx == 1){
        assert(argc >= 8);
        Timer t;

        string lod_way = argv[7];
        if (lod_way.compare("LG") == 0) load_index_LG(graph_name);
        else if (lod_way.compare("LGf") == 0) load_index_LGf(graph_name);
        else if (lod_way.compare("GRL2") == 0) load_index_GRL2(graph_name);
        else if (lod_way.compare("GRL3") == 0) load_index_GRL3(graph_name);
        else cout<<"no matching load index way."<<endl;

//        cout<<"Load index time cost = "<<integer_to_string(t.elapsed())<<endl;

    }

    assert(argc == 15);
    
    update_type = atoi(argv[8]); //0:no,  1:insert,  2:delete
    assert(update_type == 0 || update_type == 1 || update_type == 2);
    
    num_of_update = atoi(argv[9]); //10^x
    string update_algo = argv[10];
    
    if(update_type != 0) { //generate update edges...
        
        //write original index to disk for size comparison
        write_modified_index_to_disk(graph_name,0);
        
        for(ui i = 0; i < n; i++) {
            vector<ui> nei_vec;
            for(ui j = pstart[i]; j < pstart[i+1]; j++) {
                ui v = edges[j];
                nei_vec.push_back(v);
                G_map[i].insert(v);//G_map is a temporary container to make sure new edges are generated
                G_map[v].insert(i);
            }
            G_vv.push_back(nei_vec);
        }
        
        int tmp_num_of_update = num_of_update;
        if(update_type == 1) {
            while (tmp_num_of_update > 0) {
                ui a = rand()%n1;
                ui b = rand()%n2 + n1;
                assert(a >= 0 && a < n1);
                assert(b >= n1 && b < n);
                assert(G_map.find(a) != G_map.end());
                if(G_map[a].find(b) == G_map[a].end()) { //new edge
                    assert(G_map[b].find(a) == G_map[b].end());
                    G_map[a].insert(b);
                    G_map[b].insert(a);
                    new_edges_vec.push_back(make_pair(a, b));
                    -- tmp_num_of_update;
                }
            }
        }
        else {
            assert(update_type == 2);
            vector<pair<ui, ui>> all_edges;
            for(ui i = 0; i < n1; i++) {
                for(ui j = pstart[i]; j < pstart[i+1]; j++) {
                    ui v = edges[j];
                    all_edges.push_back(make_pair(i, v));
                }
            }
            
            while (tmp_num_of_update > 0 && all_edges.size() > 0) {
                ui x = rand()%all_edges.size();
                vector<pair<ui, ui>>::iterator itr = all_edges.begin();
                itr = itr + x;
                ui a = (*itr).first;
                ui b = (*itr).second;
                assert(a >= 0 && a < n1);
                assert(b >= n1 && b < n);
                new_edges_vec.push_back(make_pair(a, b));
                -- tmp_num_of_update;
                all_edges.erase(itr);
                
            }
        }
        
//        cout<<"new_edges_vec size = "<<new_edges_vec.size()<<endl;
        
        ui * c = new ui[n];
        memset(c, 0, sizeof(ui)*n);
        ui * ck = new ui[n];
        memset(ck, 0, sizeof(ui)*n);
        
        Timer T_dyn;
        if (update_algo.compare("bs") == 0) modify_graph_bs(update_type, new_edges_vec, c, rg_limit);
        else if (update_algo.compare("heu") == 0) modify_graph_heu(update_type, new_edges_vec, c, ck); //modify G_vv and index vsn
        else if (update_algo.compare("heu2") == 0) modify_graph_heu2(update_type, new_edges_vec, c); //modify G_vv and index vsn
        else cout<<"no matching update_algo."<<endl;
        cout<<"update graph and index time cost = "<<integer_to_string(T_dyn.elapsed())<<endl;
        cout<<"next, enumerate maximal similar-bicliques on the updated graph and index"<<endl;
//        cout<<"$$$$$$ total_num_idx : "<<total_num_idx<<", skipped vertex : "<<(total_num_idx - unskipped_num_idx)<<endl;
        
        //write the modified vsn to disk. (just for index size evaluation, since the modified graph is ignored)
        write_modified_index_to_disk(graph_name, 1);
        
        //transfer the modified graph in G_vv to psatrt and edges
        delete [] edges;
        m = 0;
        for(ui i = 0; i < n; i++) {
            unsigned long s = G_vv[i].size();
            m += s;
            TMPdeg[i] = s;
            degree[i] = s;
        }
        edges = new ui[m];
        pstart[0] = 0;
        for(ui i = 0; i < n; i++) {
            ui pos = pstart[i];
            for(auto e : G_vv[i]) edges[pos++] = e;
            pstart[i+1] = pos;
        }

        delete [] c;
        delete [] ck;
    }

    vr_way = atoi(argv[11]); //1 2 3 ...
    epsilon = atof(argv[12]);
    assert(epsilon >= 0 && epsilon <= 1);
    tau = atoi(argv[13]);
    assert(tau >= 1);
    cout<<"epsilon = "<<epsilon<<", tau = "<<tau<<endl;
    noRSim = atoi(argv[14]);
    Timer t;
    if (noRSim == 2) MDBC_Enum_noRSim_adv();
    else {
        cout<<"no matching noRSim!"<<endl; exit(1);
    }
        
    cout<<" - - - - - - - - - - - - - - - - - - - - - - - - -  "<<endl;
    if(over_time_flag) cout<<"| ###### OVER TIME ######"<<endl;
    cout<<"| results size = "<<MDBC_num<<endl;
    if(MDBC_num != 0){
        cout<<"| min MDBC size = "<<min_MDBC_size<<endl;
        cout<<"| max MDBC size = "<<max_MDBC_size<<endl;
    }
    cout<<"| Time cost (without I/O) = "<<integer_to_string(t.elapsed())<<endl;
    cout<<" - - - - - - - - - - - - - - - - - - - - - - - - -  "<<endl;

#ifdef _CheckResults_
//    check_results(results);
//    print_size_distribution(results);
//    compute_strength(results);
#endif
        
#ifdef _PrintResults_
    print_results();
#endif
    
#ifdef _CaseStudy_
    case_study();
#endif
    
    dele_memo();
    
    return 0;
}
