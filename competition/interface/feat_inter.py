__author__ = 'songquanwang'
import abc


class FeatInter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gen_feat(self):
        return

    @abc.abstractmethod
    def gen_feat_cv(self):
        return

    @abc.staticmothod
    def extract_feats(single_feat_path, combined_feat_path, feat_names, mode):
        return

    @abc.staticmothod
    def extract_feats_cv(feat_names, feat_path_name):
        return
