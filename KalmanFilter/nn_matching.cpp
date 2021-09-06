#include "nn_matching.h"
#include <iostream>
using namespace Eigen;

NearNeighborDisMetric::NearNeighborDisMetric(
    NearNeighborDisMetric::METRIC_TYPE metric,
    float matching_threshold, int budget)
{
  if(metric == euclidean)
    {
      _metric = &NearNeighborDisMetric::_nneuclidean_distance;
    } else if (metric == cosine)
    {
      _metric = &NearNeighborDisMetric::_nncosine_distance;
    }

  this->mating_threshold = matching_threshold;
  this->budget = budget;
  this->samples.clear();
}

DYNAMICM
NearNeighborDisMetric::distance(
    const FEATURESS &features,
    const std::vector<int>& targets)
{
  DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
const int nCore = omp_get_num_procs();
#pragma omp parallel for num_threads(nCore)
  for(int i=0; i<targets.size();i++) {
      cost_matrix.row(i) = (this->*_metric)(this->samples[targets[i]], features);
  }
  return cost_matrix;
}

void
NearNeighborDisMetric::partial_fit(
    std::vector<TRACKER_DATA> &tid_feats,
    std::vector<int> &active_targets)
{
  /*python code:
 * let feature(target_id) append to samples;
 * && delete not comfirmed target_id from samples.
 * update samples;
*/

for(int i=0; i<tid_feats.size(); i++) {
    int track_id = tid_feats[i].first;
    FEATURESS newFeatOne = tid_feats[i].second;
    if(samples.find(track_id) == samples.end()) {
          samples[track_id] = newFeatOne;
    }//add features;
}
const int nCore = omp_get_num_procs();
#pragma omp parallel for num_threads(nCore)
for(int i=0; i<tid_feats.size(); i++) {
    int track_id = tid_feats[i].first;
    FEATURESS newFeatOne = tid_feats[i].second;

    if(samples.find(track_id) != samples.end()) {//append
        int oldSize = samples[track_id].rows();
        int addSize = newFeatOne.rows();
        int newSize = oldSize + addSize;

        if(newSize <= this->budget) {
            FEATURESS newSampleFeatures(newSize, FEATURE_DIMS);
            newSampleFeatures.block(0,0, oldSize, FEATURE_DIMS) = samples[track_id];
            newSampleFeatures.block(oldSize, 0, addSize, FEATURE_DIMS) = newFeatOne;
            samples[track_id] = newSampleFeatures;
        } else {
            if(oldSize < this->budget) {//original space is not enough;
                FEATURESS newSampleFeatures(this->budget, FEATURE_DIMS);
                if(addSize >= this->budget) {
                    newSampleFeatures = newFeatOne.block(0, 0, this->budget, FEATURE_DIMS);
                } else {
                    newSampleFeatures.block(0, 0, this->budget-addSize, FEATURE_DIMS) =
                        samples[track_id].block(addSize-1, 0, this->budget-addSize, FEATURE_DIMS).eval();
                    newSampleFeatures.block(this->budget-addSize, 0, addSize, FEATURE_DIMS) = newFeatOne;
                }
                samples[track_id] = newSampleFeatures;
            } else {//original space is ok;
                if(addSize >= this->budget) {
                    samples[track_id] = newFeatOne.block(0,0, this->budget, FEATURE_DIMS);
                } else {
                    samples[track_id].block(0, 0, this->budget-addSize, FEATURE_DIMS) =
                        samples[track_id].block(addSize-1, 0, this->budget-addSize, FEATURE_DIMS).eval();                         
                    samples[track_id].block(this->budget-addSize, 0, addSize, FEATURE_DIMS) = newFeatOne;
                }
            }
        }
    }
}//add features;
    

  //erase the samples which not in active_targets;
  for(std::map<int, FEATURESS>::iterator i = samples.begin(); i != samples.end();) {
      bool flag = false;
      for(int j:active_targets) if(j == i->first) { flag=true; break; }
      if(flag == false)  samples.erase(i++);
      else i++;
    }
}

Eigen::VectorXf
NearNeighborDisMetric::_nncosine_distance(
    const FEATURESS &x, const FEATURESS &y)
{
  MatrixXf distances = _cosine_distance(x,y);
  VectorXf res = distances.colwise().minCoeff().transpose();
  return res;
}

Eigen::VectorXf
NearNeighborDisMetric::_nneuclidean_distance(
    const FEATURESS &x, const FEATURESS &y)
{
  MatrixXf distances = _pdist(x,y);
  VectorXf res = distances.colwise().maxCoeff().transpose();
  res = res.array().max(VectorXf::Zero(res.rows()).array());
  return res;
}

Eigen::MatrixXf
NearNeighborDisMetric::_pdist(const FEATURESS &x, const FEATURESS &y)
{
  int len1 = x.rows(), len2 = y.rows();
  if(len1 == 0 || len2 == 0) {
      return Eigen::MatrixXf::Zero(len1, len2);
    }
  MatrixXf res = x * y.transpose()* -2;
  res = res.colwise() + x.rowwise().squaredNorm();
  res = res.rowwise() + y.rowwise().squaredNorm().transpose();
  res = res.array().max(MatrixXf::Zero(res.rows(), res.cols()).array());
  return res;
}

Eigen::MatrixXf
NearNeighborDisMetric::_cosine_distance(
    const FEATURESS & a,
    const FEATURESS& b, bool data_is_normalized) {
  if(data_is_normalized == true) {
      //undo:
      assert(false);
    }
  MatrixXf res = 1. - (a*b.transpose()).array();
  return res;
}
