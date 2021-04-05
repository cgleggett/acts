#ifndef THRD_MSG_COL_H
#define THRD_MSG_COL_H 1

#include <mutex>
#include <sstream>
#include <thread>
#include <map>

class tid {
public:
  static int id() {
    std::lock_guard<std::mutex> lk(m_m);
    std::map<std::thread::id,int>::iterator it = m_tid.find(std::this_thread::get_id());
    if (it == m_tid.end()) {
      int sz = m_tid.size();
      m_tid[std::this_thread::get_id()] = sz+1;
      return sz+1;
    } else {
      return it->second;
    }
  }

private:
  static std::mutex m_m;
  static std::map<std::thread::id,int> m_tid;
};

std::mutex tid::m_m;
std::map<std::thread::id,int> tid::m_tid;


std::mutex c_mut;
#define MSG(msg) {\
  std::ostringstream ost; \
  std::lock_guard<std::mutex> lk(c_mut); \
  ost << "[" << tid::id() << "] " << msg; \
  std::cout << "\033[" << 29+tid::id() <<";1m" << ost.str() << "\033[m" << std::endl;	\
  }

#endif
