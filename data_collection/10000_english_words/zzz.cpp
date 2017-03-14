#include <bits/stdc++.h>
using namespace std;

void log();

fstream ptr;
string tmp;
int table[52][26];
bool flg = false;
list<string> mylist;

bool compare_nocase (const std::string& first, const std::string& second){
  
  return ( first.length() < second.length() );
}

int main() {
	
	
	ptr.open("3000words.txt",ios::in);
	//ptr.open("google-10000-english-usa.txt",ios::in);
	
	//build table
	while( !ptr.eof() ) {
		ptr >> tmp;
		cout << tmp << endl;
		if ( tmp.size() > 1 ){ 
			mylist.push_back(tmp);
			for ( int i = 0 ; i < tmp.size()-1 ; ++i ){
				if('a' <= tmp[i] && tmp[i] <= 'z') ++table[tmp[i] - 'a' ][tmp[i+1] - 'a'];
				else if('A' <= tmp[i] && tmp[i] <= 'Z') ++table[tmp[i] - 'A'+26][tmp[i+1] - 'a'];
			}
		} 
		
	}
	
	ptr.close();
	// sort words
	mylist.sort(compare_nocase);
	
	//greedy decreasing table
	for ( std::list<std::string>::iterator it=mylist.begin(); it!=mylist.end(); ) {
		
		tmp = *it;
		flg = true;
		for( int i = 0 ; i < tmp.size()-1 ; ++i ){
			if( ( 'a' <= tmp[i] && tmp[i] <= 'z') && 
				( table[tmp[i] - 'a'][tmp[i+1] - 'a'] < 2 ) ){
					flg = false;
				break;
			}
			else if(( 'A' <= tmp[i] && tmp[i] <= 'Z') && 
					( table[tmp[i] - 'A'+26][tmp[i+1] - 'a'] < 2 ) ){
					flg = false;
				break;
			}
			
		}
		
		if ( flg ) {
			for( int i = 0 ; i < tmp.size()-1 ; ++i ){
				if('a' <= tmp[i] && tmp[i] <= 'z') --table[tmp[i] - 'a' ][tmp[i+1] - 'a'];
				else if('A' <= tmp[i] && tmp[i] <= 'Z') --table[tmp[i] - 'A'+26][tmp[i+1] - 'a'];
			}
			++it;
			mylist.remove(tmp);
		}
		else {
			++it;
		}
	}
	
	log();
	ptr.open("important_words.txt",ios::out);
	for ( std::list<std::string>::iterator it=mylist.begin(); it!=mylist.end(); ++it ) {
		ptr << *it << endl;
	} 
	
	
	return 0;
} 

void log() {
	printf("    ");
	for( int i = 0 ; i < 26 ; ++i )
		printf("%4c",'a'+i);
		putchar('\n');
		
	for( int i = 0 ; i < 52 ; ++i ){
		if(i < 26)printf("%4c",'a'+i);
		else printf("%4c",'A'+i-26);
		
		for( int j = 0 ; j < 26 ; ++j )
			printf("%4d",table[i][j]);
		putchar('\n');
	}
}
