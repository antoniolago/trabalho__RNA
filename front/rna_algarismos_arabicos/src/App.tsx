import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import ImageEditor from './ImageEditor'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <ImageEditor />
    </>
  )
}

export default App
