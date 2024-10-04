/* timer tools for code module or subprograms */

#pragma once

#include <chrono>
#include <string>
#include <fstream>
#include <ctime>
#include <map>
#include <initializer_list>
#include <iomanip>
#include <type_traits>
#include "dense_base.h"

namespace KylinVib
{
    using std::map;
    using std::string_view;    
    using std::clock_t;
    using std::clock;
    using std::ostream;
    using std::setw;
    using std::setfill;
    using std::setprecision;
    namespace Watson
    {
        struct Timer
        {
            /* cpu time cost of every module */
            map<string_view,double> CpuTimes;

            /* wall time cost of every module */
            map<string_view,double> WallTimes;

            /* start point of cpu */
            clock_t CStart;

            /* end point of cpu */
            clock_t CEnd;

            /* start point of cpu */
            std::chrono::time_point<std::chrono::high_resolution_clock> WStart;

            /* end point of cpu */
            std::chrono::time_point<std::chrono::high_resolution_clock> WEnd;
        };

        // start a module
        void start_module(Timer & tm, const char * m)
        {
            string_view module(m);
            if(tm.CpuTimes.find(module) == tm.CpuTimes.end())
            {
                tm.CpuTimes[module] = 0.0;
                tm.WallTimes[module] = 0.0;
            }
            tm.CStart = clock();
            tm.WStart = std::chrono::high_resolution_clock::now();
        }

        // end a module
        void end_module(Timer & tm, const char * m)
        {
            string_view module(m);
            if(tm.CpuTimes.find(module) == tm.CpuTimes.end())
            {
                cout << "No module named " << module << "!" << endl;
                exit(1);
            }
            tm.CEnd = clock();
            tm.WEnd = std::chrono::high_resolution_clock::now();
            tm.CpuTimes[module] += 1000.0 * (tm.CEnd - tm.CStart) / CLOCKS_PER_SEC;
            tm.WallTimes[module] += std::chrono::duration<double,std::milli>(tm.WEnd-tm.WStart).count();
        }

        // print message
        ostream & operator<<(ostream & os, Timer & tm)
        {
            os << std::left << setw(15) << "Modules"
            << setw(15) << "Cpu times/ms"
            << setw(15) << "Wall times/ms"
            << setw(15) << "Percentage" << endl;
            
            double TotalWall = 0.0;
            for(auto it=tm.WallTimes.begin();it!=tm.WallTimes.end();++it)
            {
                    TotalWall += it->second;
            }
            for(auto it=tm.WallTimes.begin();it!=tm.WallTimes.end();++it)
            {
                os << std::left << setw(15) << it->first
                    << setw(15) << setprecision(2) << std::scientific << tm.CpuTimes[it->first]
                    << setw(15) << setprecision(2) << std::scientific << tm.WallTimes[it->first]
                    << setw(15) << setprecision(2) << std::fixed << tm.WallTimes[it->first] / TotalWall * 100.0
                    << endl;
            }
            return os;
        }
    }
} 
