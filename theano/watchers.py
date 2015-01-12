from collections import namedtuple
import time
import cPickle

class Trainable(object):
    def obj(self):
        """ Return objective expression """
        raise NotImplemented

    def params(self):
        """ Return a list of trainable params """
        raise NotImplemented

    def args(self):
        """ Return training arguments """
        raise NotImplemented

WatcherInfo = namedtuple("WatcherInfo", "iter time_elapsed loss end")

class Optimizer(object):
    def __init__(self):
        self.watchers = []
        self.watcher_info = WatcherInfo(0,0,0, False)

    def addWatcher(self, watcher):
        self.watchers.append(watcher)

    def triggerWatchers(self):
        for watcher in self.watchers:
            if watcher.trigger(self.watcher_info):
                watcher.action(self.watcher_info)

    def optimize(self, n_iters):
        time_start = time.time()
        for i in range(n_iters):
            self.optimize_iter(i)
            self.watcher_info = WatcherInfo(i, time.time()-time_start, self.get_loss(i), i==n_iters-1)
            self.triggerWatchers()

    def optimize_iter(self, i):
        raise NotImplementedError
    
    def get_loss(self, i):
        raise NotImplementedError

class Watcher(object):
    def __init__(self, condition):
        self.condition = condition

    def trigger(self, watcher_info):
        return self.condition(watcher_info)

    def action(self, watcher_info):
        raise NotImplementedError

class InfoWatcher(Watcher):
    def __init__(self, condition):
        super(InfoWatcher, self).__init__(condition)

    def action(self, winfo):
        print "[Info:%s] Iter:%d Time:%f Loss:%f" % (self.condition, winfo.iter, winfo.time_elapsed, winfo.loss)

class TimeWatcher(Watcher):
    def __init__(self, condition):
        super(TimeWatcher, self).__init__(condition)

    def action(self, winfo):
        print "[Time:%s] Time Elapsed:%f" % (self.condition, winfo.time_elapsed)

class PickleWatcher(Watcher):
    def __init__(self, to_pickle, name_format, condition):
        super(PickleWatcher, self).__init__(condition)
        self.to_pickle = to_pickle
        self.name_format = name_format

    def action(self, winfo):
        fname = self.name_format
        print "[Save:%s] Saving weights to %s" % (self.condition, fname)
        with open(fname, 'wb') as pklfile:
            cPickle.dump(self.to_pickle, pklfile)

# Watcher Conditions

class OnIter(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, watcher_info):
        return watcher_info.iter % self.n == 0

    def __str__(self):
        return 'OnIter{%s}' % self.n

class OnEnd(object):
    def __call__(self, watcher_info):
        return watcher_info.end

    def __str__(self):
        return 'OnEnd'

class OnTime(object):
    def __init__(self, n):
        self.n = n
        self.last_time = 0

    def __call__(self, watcher_info):
        time = watcher_info.time_elapsed
        if time-self.last_time > self.n:
            self.last_time += self.n
            return True
        return False

    def __str__(self):
        return 'OnTime{%s}' % self.n

class FOptimizer(Optimizer):
    def __init__(self, f, *args, **kwargs):
        super(FOptimizer, self).__init__()
        self.optimizer = f(*args, **kwargs)
        self.loss = 0

    def optimize_iter(self, i):
        self.loss = self.optimizer()

    def get_loss(self, i):
        return self.loss[0]

