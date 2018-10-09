#pragma once
#include <memory>

#include "common/definitions.h"
#include "data/alignment.h"

namespace marian {

typedef std::tuple<size_t, float> WordWithScore;
typedef std::vector<WordWithScore> AlternativeWordsWithScores;
typedef std::vector<AlternativeWordsWithScores> TranslationsWithAlternativeWordsWithScores;

class Hypothesis {
public:
  Hypothesis() : prevHyp_(nullptr), prevIndex_(0), word_(0), pathScore_(0.0) {}

  Hypothesis(const Ptr<Hypothesis> prevHyp,
             size_t word,
             size_t prevIndex,
             float pathScore,
             const std::vector<unsigned int> topWordScoresKeys,
             const std::vector<float> topWordScoresVals)
      : prevHyp_(prevHyp), prevIndex_(prevIndex), word_(word), pathScore_(pathScore), topWordScoresKeys_(topWordScoresKeys), topWordScoresVals_(topWordScoresVals) {}

  const Ptr<Hypothesis> GetPrevHyp() const { return prevHyp_; }

  size_t GetWord() const { return word_; }

  size_t GetPrevStateIndex() const { return prevIndex_; }

  float GetPathScore() const { return pathScore_; }

  std::vector<float>& GetScoreBreakdown() { return scoreBreakdown_; }
  std::vector<float>& GetAlignment() { return alignment_; }

  void SetAlignment(const std::vector<float>& align) { alignment_ = align; };

  // helpers to trace back paths referenced from this hypothesis
  Words TracebackWords()
  {
      Words targetWords;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          targetWords.push_back(hyp->GetWord());
          // std::cerr << hyp->GetWord() << " " << hyp << std::endl;
      }
      std::reverse(targetWords.begin(), targetWords.end());
      return targetWords;
  }

  // get soft alignments for each target word starting from the hyp one
  typedef data::SoftAlignment SoftAlignment;
  SoftAlignment TracebackAlignment()
  {
      SoftAlignment align;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          align.push_back(hyp->GetAlignment());
      }
      std::reverse(align.begin(), align.end());
      return align;
  }

  // get translations with the top k alternative words at each position with log-probability
  TranslationsWithAlternativeWordsWithScores TracebackTranslationsWithAlternativeWordsWithScores()
  {
      TranslationsWithAlternativeWordsWithScores rv;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          AlternativeWordsWithScores currAltWwS;
          for (size_t j = 0; j < hyp->topWordScoresKeys_.size(); ++j) {
            WordWithScore wwS(hyp->topWordScoresKeys_[j], hyp->topWordScoresVals_[j]);
            currAltWwS.push_back(wwS);
          }
          rv.push_back(currAltWwS);
          // std::cerr << rv << " " << hyp << std::endl;
      }
      std::reverse(rv.begin(), rv.end());
      return rv;
  }

private:
  const Ptr<Hypothesis> prevHyp_;
  const size_t prevIndex_;
  const size_t word_;
  const float pathScore_;

  std::vector<float> scoreBreakdown_;
  std::vector<float> alignment_;

  std::vector<unsigned int> topWordScoresKeys_;
  std::vector<float> topWordScoresVals_;
};

typedef std::vector<Ptr<Hypothesis>> Beam;                // Beam = vector of hypotheses
typedef std::vector<Beam> Beams;                          // Beams = vector of vector of hypotheses
typedef std::vector<size_t> Words;                        // Words = vector of word ids
typedef std::tuple<Words, Ptr<Hypothesis>, float> Result; // (word ids for hyp, hyp, normalized sentence score for hyp)
typedef std::vector<Result> NBestList;                    // sorted vector of (word ids, hyp, sent score) tuples

}  // namespace marian
