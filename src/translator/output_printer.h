#pragma once

#include <vector>

#include "common/config.h"
#include "common/utils.h"
#include "data/alignment.h"
#include "data/vocab.h"
#include "translator/history.h"
#include "translator/hypothesis.h"

namespace marian {

class OutputPrinter {
public:
  OutputPrinter(Ptr<Config> options, Ptr<Vocab> vocab)
      : vocab_(vocab),
        reverse_(options->get<bool>("right-left")),
        nbest_(options->get<bool>("n-best", false)
                   ? options->get<size_t>("beam-size")
                   : 0),
        alignment_(options->get<std::string>("alignment", "")),
        alignmentThreshold_(getAlignmentThreshold(alignment_)),
        outputWordScores_(options->get<size_t>("output-word-scores") > 0) {}

  template <class OStream>
  void print(Ptr<History> history, OStream& best1, OStream& bestn) {
    const auto& nbl = history->NBest(nbest_);

    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& words = std::get<0>(result);
      const auto& hypo = std::get<1>(result);

      std::string translation = utils::Join((*vocab_)(words), " ", reverse_);
      bestn << history->GetLineNum() << " ||| " << translation;

      if(!alignment_.empty())
        bestn << " ||| " << getAlignment(hypo);

      bestn << " |||";
      if(hypo->GetScoreBreakdown().empty()) {
        bestn << " F0=" << hypo->GetPathScore();
      } else {
        for(size_t j = 0; j < hypo->GetScoreBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->GetScoreBreakdown()[j];
        }
      }

      float realScore = std::get<2>(result);
      bestn << " ||| " << realScore;

      if (outputWordScores_) {
        OutputWordScores(hypo, bestn);
      }

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }

    auto result = history->Top();
    const auto& words = std::get<0>(result);

    std::string translation = utils::Join((*vocab_)(words), " ", reverse_);

    best1 << translation;
    if(!alignment_.empty()) {
      const auto& hypo = std::get<1>(result);
      best1 << " ||| " << getAlignment(hypo);
    }

    if (outputWordScores_) {
      OutputWordScores(std::get<1>(result), best1);
    }

    best1 << std::flush;
  }

private:
  Ptr<Vocab> vocab_;
  bool reverse_{false};
  size_t nbest_{0};
  std::string alignment_;
  float alignmentThreshold_{0.f};
  bool outputWordScores_{false};

  std::string getAlignment(const Ptr<Hypothesis>& hyp);

  float getAlignmentThreshold(const std::string& str) {
    try {
      return std::max(std::stof(str), 0.f);
    } catch(...) {
      return 0.f;
    }
  }

  template <class OStream>
  void OutputWordScores(Ptr<Hypothesis> hypo, OStream& ostream)
  {
    auto& vocabRef = *(vocab_);
    ostream << " ||| word_scores: ";
    auto translAltWordsScores = hypo->TracebackTranslationsWithAlternativeWordsWithScores();
    for (size_t i = 0; i < translAltWordsScores.size(); ++i) {
      auto curAltWordsScores = translAltWordsScores[i];
      ostream << "[ ";
      for (size_t j = 0; j < curAltWordsScores.size(); ++j) {
        auto wordScore = curAltWordsScores[j];
        ostream << vocabRef[std::get<0>(wordScore)] << " " << std::get<1>(wordScore) << " ";
      }
      ostream << "]";
    }
  }
};
}  // namespace marian
