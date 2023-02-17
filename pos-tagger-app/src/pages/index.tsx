import Head from 'next/head'
import styled from 'styled-components'

const Home = () => {
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
        <TagContainer>
        <TextInput/>
        </TagContainer>
      </MainContainer>
    </>
  )
}

const MainContainer = styled.main`
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  padding: 2rem 0;
  height: 100vh;
  width: 100vw;
  background-color: white;
`

const Title = styled.div`
  font-family: 'Bitter';
  font-size: 2em;
  padding: 36px 0;
`

const TagContainer = styled.div`
  display: flex;
  width: 45%;
  height: 35%;
`

const TextInput = styled.textarea`
  font-family: 'Open Sans';
  font-size: 1em;
  width: 100%;
  min-width: 100%;
  height: 100%;
  min-height: 60%;
  max-height: 100%;
  padding: 20px;
  line-height: 1.6;
`

export default Home;
