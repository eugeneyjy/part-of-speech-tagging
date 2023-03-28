import React, { FC, ReactElement } from 'react';
import Head from 'next/head';
import styled from 'styled-components';
import { tagColors } from '@/styles/color-theme';

interface TagType {
  name: string
}

interface DefProps {
  TagClickHandler: (event: React.MouseEvent<HTMLSpanElement>) => void
}

type PredictData = {
  tokens: Array<string>,
  tags: Array<string>
}

const fakePredictData: PredictData = {
  tokens: ['this', 'is', 'a', 'part-of-speech', '(', 'pos', ')', 'tagger', 'trained', 'using', 'bi-lstm', 'model', '.', 'enter', 'sentences', 'right', 'in', 'here', 'and', 'they', 'will', 'be', 'color', 'coded', 'with', 'their', 'respective', 'pos', 'tag', 'in', 'real', 'time', '.', 'check', 'below', 'for', 'the', 'definition', 'of', 'each', 'tags', '.'],
  tags: ['PRON', 'AUX', 'DET', 'NOUN', 'PUNCT', 'NOUN', 'PUNCT', 'PROPN', 'VERB', 'VERB', 'NOUN', 'NOUN', 'PUNCT', 'VERB', 'NOUN', 'ADV', 'ADP', 'ADV', 'CCONJ', 'PRON', 'AUX', 'AUX', 'VERB', 'NOUN', 'ADP', 'PRON', 'ADJ', 'NOUN', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'PUNCT', 'VERB', 'ADV', 'ADP', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PUNCT']
}

const Home = () => {
  const [tagName, setTagName] = React.useState<string>('Adjective');
  const [inputText, setInputText] = React.useState<string>(`This is a Part-of-speech (POS) tagger trained using Bi-LSTM model. \
Enter sentences right in here and they will be color coded with their respective POS tag in real time. \
Check below for the definition of each tags.`);
  const [backdropText, setbackdropText] = React.useState<ReactElement[]>([]);

  const switchDefinition = (event: React.MouseEvent<HTMLSpanElement>) => {
    event.preventDefault();
    const target = event.currentTarget;
    const name = target.textContent;
    let validatedName: string = typeof name === 'string' ?  name : '';
    setTagName(validatedName);
  };

  const updateInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    event.preventDefault();
    let input = event.currentTarget.value
    setInputText(input);
  }

  const populateHighlight = (input:string, predictData: PredictData) => {
    let highlightedText:ReactElement[] = [];
    if (predictData.tags.length > 0 && predictData.tokens.length > 0) {
      let currIdx = 0;
      let currTagIdx = 0;
      while (currIdx < input.length) {
        if (input[currIdx] === ' ') {
          highlightedText.push(<React.Fragment key={currIdx}> </React.Fragment>);
          currIdx += 1;
        } else if (input[currIdx] === '\n'){
          highlightedText.push(<React.Fragment key={currIdx}><br/></React.Fragment>);
          currIdx += 1;
        } else {
          let tokenLen = predictData.tokens[currTagIdx].length;
          let token = input.slice(currIdx, currIdx+tokenLen);
          const tag = <TagMark key={currIdx} name={predictData.tags[currTagIdx].toLowerCase()}>{token}</TagMark>;
          highlightedText.push(tag);
          currIdx += tokenLen;
          currTagIdx += 1;
        }
      }
    }
    setbackdropText(highlightedText);
  }

  React.useEffect(() => {
    const requestOptions = {
      method: 'POST',
      body: JSON.stringify({"sentence": inputText}),
      headers: {"Content-Type": "application/json"}
    }
    fetch(`http://localhost:5000/tag`, requestOptions)
      .then((res) => {
        if (res.status == 200) {
          res.json().then(data => {
            let predictData: PredictData = {
              tokens: data.tokens,
              tags: data.tags
            };
            populateHighlight(inputText, predictData);
          })
        }
      });
  }, [inputText]);

  const tagMap = new Map<string, ReactElement>();
  tagMap.set('Adjective', <AdjDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Adposition', <AdpDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Adverb', <AdvDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Auxillary', <AuxDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Coordinating conjunction', <CconjDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Determiner', <DetDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Interjection', <IntjDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Noun', <NounDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Numeral', <NumDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Particle', <PartDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Pronoun', <PronDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Proper noun', <PropnDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Punctuation', <PunctDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Subordinating conjunction', <SconjDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Symbol', <SymDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Verb', <VerbDef TagClickHandler={switchDefinition}/>);
  tagMap.set('Other', <XDef TagClickHandler={switchDefinition}/>);

  return (
    <>
      <Head>
        <title>Part-Of-Speech Tagger</title>
        <meta name="description" content="A part-of-speech tagger utilizing BiLSTM model" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <MainContainer>
        <Title>
            Part-Of-Speech Tagger
        </Title>
        <ContentContainer>
          <InputContainer>
            <BackdropContainer>
              <HighlightContainer>
                {backdropText}
              </HighlightContainer>
            </BackdropContainer>
            <TextInput value={inputText} onChange={updateInput}/>
          </InputContainer>
          <InfoContainer>
            <TagsContainer>
              <Tag onClick={switchDefinition} name='adj'>Adjective</Tag>
              <Tag onClick={switchDefinition} name='adp'>Adposition</Tag>
              <Tag onClick={switchDefinition} name='adv'>Adverb</Tag>
              <Tag onClick={switchDefinition} name='aux'>Auxillary</Tag>
              <Tag onClick={switchDefinition} name='cconj'>Coordinating conjunction</Tag>
              <Tag onClick={switchDefinition} name='det'>Determiner</Tag>
              <Tag onClick={switchDefinition} name='intj'>Interjection</Tag>
              <Tag onClick={switchDefinition} name='noun'>Noun</Tag>
              <Tag onClick={switchDefinition} name='num'>Numeral</Tag>
              <Tag onClick={switchDefinition} name='part'>Particle</Tag>
              <Tag onClick={switchDefinition} name='pron'>Pronoun</Tag>
              <Tag onClick={switchDefinition} name='propn'>Proper noun</Tag>
              <Tag onClick={switchDefinition} name='punc'>Punctuation</Tag>
              <Tag onClick={switchDefinition} name='sconj'>Subordinating conjunction</Tag>
              <Tag onClick={switchDefinition} name='sym'>Symbol</Tag>
              <Tag onClick={switchDefinition} name='verb'>Verb</Tag>
              <Tag onClick={switchDefinition} name='other'>Other</Tag>
            </TagsContainer>
            <DefinitionContainer>
              <DefinitionTitle>
                {tagName}
              </DefinitionTitle>
              <>
              {tagMap.get(tagName)}
              </>
            </DefinitionContainer>
          </InfoContainer>
        </ContentContainer>
      </MainContainer>
    </>
  );
};

const MainContainer = styled.main`
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  padding: 2rem 0;
  height: 100vh;
  width: 100vw;
  background-color: white;
`;

const Title = styled.div`
  font-family: 'Bitter';
  font-size: 2em;
  padding: 36px 0;
`;

const ContentContainer = styled.div`
  display: flex;
  flex-direction: column;
  width: 45%;
  height: 100%;
`;

const InputContainer = styled.div`
  position: relative;
  width: 100%;
  height: 40%;
  min-height: 250px;
  margin: 0 auto;
`;

const BackdropContainer = styled.div`
  position: absolute;
  z-index: 1;
  width: 100%;
  height: 100%;
  border: 2px solid #685972;
  background-color: #fff;
  overflow: auto;
`;

const TextInput = styled.textarea`
  position: absolute;
  z-index: 2;
  font-family: 'Open Sans';
  font-size: 1.1em;
  width: 100%;
  height: 100%;
  white-space: pre-wrap;
	word-wrap: break-word;
  margin: 0;
  border: 2px solid #685972;
  background-color: transparent;
  overflow: auto;
  padding: 20px;
  line-height: 1.6;
  resize: none;
  &:focus {
    outline: none;
    box-shadow: 0px 0px 2px purple;
}
`;

const HighlightContainer = styled.div`
  font-family: 'Open Sans';
  font-size: 1.1em;
	white-space: pre-wrap;
	word-wrap: break-word;
	color: transparent;
  padding: 20px;
  line-height: 1.6;
  resize: none;
`;

const InfoContainer = styled.div`
  width: 100%;
  height: 100%;
`;

const TagsContainer = styled.div`
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 10px;
  width: 100%;
  padding: 20px 0;
`;

const Tag = styled.span<TagType>`
  font-family: 'Bitter';
  padding: 3px 10px;
  background-color: ${props => tagColors[props.name]};
  height: fit-content;
  cursor: pointer;
  user-select: none;
`;

const TagMark = styled.mark<TagType>`
  position: 'absolute';
  color: transparent;
  background-color ${props => tagColors[props.name]};
`

const DefinitionContainer = styled.div`
  display: flex;
  flex-direction: column;
`;

const DefinitionTitle = styled.div`
  font-family: 'Bitter';
  font-size: 1.5em;
`;

const Definition = styled.div`
  white-space: pre-line;
`;


const AdjDef: FC<DefProps> = (props) => {
  const name = 'adj';
  return(
    <Definition>
      Adjectives are words that typically modify nouns and specify their properties or attributes:<br/>
      The <Tag name={name}>oldest</Tag> French bridge<br/>
      They may also function as predicates, as in:<br/>
      The car is <Tag name={name}>green</Tag>.<br/>
      Some words that could be seen as adjectives (and are tagged as such in other annotation schemes) have a different tag in Universal Dependencies (UD): 
      See <Tag name='det' onClick={props.TagClickHandler}>Determiner</Tag> and <Tag name='num' onClick={props.TagClickHandler}>Numeral</Tag>.<br/>
      Adjective is also used for “proper adjectives” such as European (“proper” as in proper nouns, i.e., words that are derived from names but are adjectives rather than nouns).<br/>
  </Definition>
  );
};

const AdpDef: FC<DefProps> = (props) => {
  return(
    <Definition>
      Adposition is a cover term for prepositions and postpositions. 
      Adpositions belong to a closed set of items that occur before (preposition) or after (postposition) a complement composed of a noun phrase, 
      noun, pronoun, or clause that functions as a noun phrase, and that form a single structure with the complement to express its grammatical 
      and semantic relation to another unit within a clause.<br/>

      In many languages, adpositions can take the form of fixed multiword expressions, such as in spite of, because of, thanks to. 
      The component words are then still tagged according to their basic use (in is ADP, spite is NOUN, etc.) and their status 
      as multiword expressions are accounted for in the syntactic annotation.
    </Definition>
  );
};

const AdvDef: FC<DefProps> = (props) => {
  return(
    <Definition>
      Adverbs are words that typically modify verbs for such categories as time, place, direction or manner. 
      They may also modify adjectives and other adverbs, as in very briefly or arguably wrong.<br/>

      There is a closed subclass of pronominal adverbs that refer to circumstances in context, rather than naming them directly; 
      similarly to pronouns, these can be categorized as interrogative, relative, demonstrative etc. 
      Pronominal adverbs also get the ADV part-of-speech tag but they are differentiated by additional features.
    </Definition>
  );
};

const AuxDef: FC<DefProps> = (props) => {
  return(
    <Definition>
      An auxiliary is a function word that accompanies the lexical verb of a verb phrase and expresses grammatical distinctions not 
      carried by the lexical verb, such as person, number, tense, mood, aspect, voice or evidentiality. It is often a verb 
      (which may have non-auxiliary uses as well) but many languages have nonverbal TAME markers and these should also be tagged AUX. 
      The class AUX also include copulas (in the narrow sense of pure linking words for nonverbal predication).<br/>

      Modal verbs may count as auxiliaries in some languages (English). 
      In other languages their behavior is not too different from the main verbs and they are thus tagged VERB.<br/>

      Note that not all languages have grammaticalized auxiliaries, and even where they exist the dividing line between 
      full verbs and auxiliaries can be expected to vary between languages. 
      Exactly which words are counted as AUX should be part of the language-specific documentation.
    </Definition>
  );
};

const CconjDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A coordinating conjunction is a word that links words or larger constituents without syntactically 
      subordinating one to the other and expresses a semantic relationship between them. <br/>

      For subordinating conjunctions, see SCONJ.
    </Definition>
  );
};

const DetDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      Determiners are words that modify nouns or noun phrases and express the reference of the noun phrase in context. 
      That is, a determiner may indicate whether the noun is referring to a definite or indefinite element of a class, 
      to a closer or more distant element, to an element belonging to a specified person or thing, to a particular number or quantity, etc.<br/>

      Determiners under this definition include both articles and pro-adjectives (pronominal adjectives), 
      which is a slightly broader sense than what is usually regarded as determiners in English. In particular, 
      there is no general requirement that a nominal can be modified by at most one determiner, 
      although some languages may show a strong tendency towards such a constraint. (For example, 
      an English nominal usually allows only one DET modifier, but there are occasional cases of addeterminers, 
      which appear outside the usual determiner, such as [en] all in all the children survived. In such cases, both all and the are given the POS DET.)<br/>

      Note that the DET tag includes (pronominal) quantifiers (words like many, few, several), 
      which are included among determiners in some languages but may belong to numerals in others. 
      However, cardinal numerals in the narrow sense (one, five, hundred) are not tagged DET even though some 
      authors would include them in quantifiers. Cardinal numbers have their own tag NUM.<br/>

      Also note that the notion of determiners is unknown in traditional grammar of some languages (e.g. Czech); 
      words equivalent to English determiners may be traditionally classified as pronouns and/or numerals in these languages. 
      In order to annotate the same thing the same way across languages, the words satisfying our definition of determiners should be tagged DET in these languages as well.<br/>

      It is not always crystal clear where pronouns end and determiners start. 
      Unlike in UD v1 it is no longer required that they are told apart solely on the base of the context. 
      The words can be pre-classified in the dictionary as either PRON or DET, based on their typical syntactic distribution 
      (and morphology, when applicable). Language-specific documentation should list all determiners (it is a closed class) and point out ambiguities, if any.
    </Definition>
  );
};

const IntjDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A coordinating conjunction is a word that links words or larger constituents without syntactically 
      subordinating one to the other and expresses a semantic relationship between them. <br/>

      For subordinating conjunctions, see SCONJ.
    </Definition>
  );
};

const NounDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      Nouns are a part of speech typically denoting a person, place, thing, animal or idea.<br/>

      The NOUN tag is intended for common nouns only. See PROPN for proper nouns and PRON for pronouns.<br/>

      Note that some verb forms such as gerunds and infinitives may share properties and usage of nouns and verbs. 
      Depending on language and context, they may be classified as either VERB or NOUN.
    </Definition>
  );
};

const NumDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A numeral is a word, functioning most typically as a determiner, adjective or pronoun, 
      that expresses a number and a relation to the number, such as quantity, sequence, frequency or fraction.<br/>

      Note that cardinal numerals are covered by NUM whether they are used as determiners or not (as in Windows Seven) 
      and whether they are expressed as words (four), digits (4) or Roman numerals (IV). Other words functioning as determiners 
      (including quantifiers such as many and few) are tagged DET.<br/>

      Note that there are words that may be traditionally called numerals in some languages (e.g. Czech) but which are not tagged NUM. 
      Such non-cardinal numerals belong to other parts of speech in our universal tagging scheme, based mainly on syntactic criteria: 
      ordinal numerals are adjectives (first, second, third) or adverbs ([cs] poprvé “for the first time”), multiplicative numerals are adverbs (once, twice) etc.<br/>

      Word tokens consisting of digits and (optionally) punctuation characters are generally considered cardinal numbers and tagged as NUM. 
      This includes numeric date/time formats (11:00) and phone numbers. Words mixing digits and alphabetic characters should, however, ordinarily be excluded. 
      In English, for example, pluralized numbers (the 1970s, the seventies) are treated as plural NOUNs, 
      while mixed alphanumeric street addresses (221B) and product names (130XE) are PROPN.
    </Definition>
  );
};

const PartDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      Particles are function words that must be associated with another word or phrase to impart meaning and 
      that do not satisfy definitions of other universal parts of speech (e.g. adpositions, coordinating conjunctions, subordinating conjunctions or auxiliary verbs). 
      Particles may encode grammatical categories such as negation, mood, tense etc. Particles are normally not inflected, although exceptions may occur.<br/>

      Note that the PART tag does not cover so-called verbal particles in Germanic languages, as in give in or end up. 
      These are adpositions or adverbs by origin and are tagged accordingly ADP or ADV. Separable verb prefixes in German are treated analogically.<br/>

      Note that not all function words that are traditionally called particles in Japanese automatically qualify for the PART tag. 
      Some of them do, e.g. the question particle か / ka. Others (e.g. に / ni, の / no) are parallel to adpositions in other languages and should thus be tagged ADP.<br/>

      In general, the PART tag should be used restrictively and only when no other tag is possible. 
      The the language-specific documentation should list the words classified as PART in the given language.
    </Definition>
  );
};

const PronDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      Pronouns are words that substitute for nouns or noun phrases, whose meaning is recoverable from the linguistic or extralinguistic context.<br/>

      Pronouns under this definition function like nouns. Note that some languages traditionally extend the term pronoun to words that substitute for adjectives. 
      Such words are not tagged PRON under our universal scheme. They are tagged as determiners in order to annotate the same thing the same way across languages.<br/>

      It is not always crystal clear where pronouns end and determiners start. Unlike in UD v1 it is no longer required that they are told apart solely on the base of the context. 
      The words can be pre-classified in the dictionary as either PRON or DET, based on their typical syntactic distribution (and morphology, when applicable). 
      Language-specific documentation should list all pronouns (it is a closed class) and point out ambiguities, if any.
    </Definition>
  );
};

const PropnDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A proper noun is a noun (or nominal content word) that is the name (or part of the name) of a specific individual, place, or object.<br/>

      Note that PROPN is only used for the subclass of nouns that are used as names and that often exhibit special syntactic properties 
      (such as occurring without an article in the singular in English). When other phrases or sentences are used as names, 
      the component words retain their original tags. For example, in Cat on a Hot Tin Roof, Cat is NOUN, on is ADP, a is DET, etc.<br/>

      A fine point is that it is not uncommon to regard words that are etymologically adjectives or participles as proper nouns when 
      they appear as part of a multiword name that overall functions like a proper noun, for example in the Yellow Pages, United Airlines or 
      Thrall Manufacturing Company. This is certainly the practice for the English Penn Treebank tag set. However, the practice should not be 
      copied from English to other languages if it is not linguistically justified there. For example, in Czech, Spojené státy “United States” is 
      an adjective followed by a common noun; their tags in UD are ADJ NOUN and the adjective modifies the noun via the amod relation.<br/>

      Acronyms of proper nouns, such as UN and NATO, should be tagged PROPN. Even if they contain numbers (as in various product names), 
      they are tagged PROPN and not SYM: 130XE, DC10, DC-10. However, if the token consists entirely of digits (like 7 in Windows 7), it is tagged NUM.
    </Definition>
  );
};

const PunctDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      Punctuation marks are non-alphabetical characters and character groups used in many languages to delimit linguistic units in printed text.<br/>

      Punctuation is not taken to include logograms such as $, %, and §, which are instead tagged as SYM. 
      (Hint: if it corresponds to a word that you pronounce, such as dollar or percent, it is SYM and not PUNCT.)<br/>

      Spoken corpora contain symbols representing pauses, laughter and other sounds; we treat them as punctuation, too. 
      In these cases it is even not required that all characters of the token are non-alphabetical. 
      One can represent a pause using a special character such as #, or using some more descriptive coding such as [:pause].
    </Definition>
  );
};

const SconjDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A subordinating conjunction is a conjunction that links constructions by making one of them a constituent of the other. 
      The subordinating conjunction typically marks the incorporated constituent which has the status of a (subordinate) clause.
    </Definition>
  );
};

const SymDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A symbol is a word-like entity that differs from ordinary words by form, function, or both.<br/>

      Many symbols are or contain special non-alphanumeric characters, similarly to punctuation. 
      What makes them different from punctuation is that they can be substituted by normal words. 
      This involves all currency symbols, e.g. $ 75 is identical to seventy-five dollars.<br/>

      Mathematical operators form another group of symbols.<br/>

      Another group of symbols is emoticons and emoji.
    </Definition>
  );
};

const VerbDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      A verb is a member of the syntactic class of words that typically signal events and actions, 
      can constitute a minimal predicate in a clause, and govern the number and types of other constituents which may occur in the clause. 
      Verbs are often associated with grammatical categories like tense, mood, aspect and voice, 
      which can either be expressed inflectionally or using auxilliary verbs or particles.<br/>

      Note that the VERB tag covers main verbs (content verbs) but it does not cover auxiliary verbs and verbal copulas 
      (in the narrow sense), for which there is the AUX tag. Modal verbs may be considered VERB or AUX, depending on their behavior in the given language. 
      Language-specific documentation should specify which verbs are tagged AUX in which contexts.
    </Definition>
  );
};

const XDef: FC<DefProps> = (props) => {
  return (
    <Definition>
      The tag X is used for words that for some reason cannot be assigned a real part-of-speech category. It should be used very restrictively.<br/>

      A special usage of X is for cases of code-switching where it is not possible (or meaningful) to analyze the intervening language grammatically 
      (and where the dependency relation flat:foreign is typically used in the syntactic analysis). 
      This usage does not extend to ordinary loan words which should be assigned a normal part-of-speech. 
      For example, in he put on a large sombrero, sombrero is an ordinary NOUN.
    </Definition>
  );
};


export default Home;
