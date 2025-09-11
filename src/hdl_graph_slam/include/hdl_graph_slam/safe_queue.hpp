#ifndef SAFE_QUEUE
#define SAFE_QUEUE

#include <ros/init.h>
#include <queue>
#include <mutex>
#include <condition_variable>

// A threadsafe-queue.
// shamelessly copied from https://stackoverflow.com/a/16075550
template<class T>
class SafeQueue {
  public:
    SafeQueue(void) : q(), m(), c() {}

    ~SafeQueue(void) {}

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue(void) {
        std::unique_lock<std::mutex> lock(m);
        while(q.empty() && ros::ok()) {
            // release lock as long as the wait and reaquire it afterwards.
            c.wait(lock);
        }
        T val = q.front();
        q.pop();
        return val;
    }

    // get the number of elements in the queue
    size_t size() {
        std::lock_guard<std::mutex> lock(m);
        return q.size();
    }

    // ping the sleeping consumer thread to check ros::ok and terminate
    void wakeup() {
        c.notify_all();
    }

  private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
};
#endif
