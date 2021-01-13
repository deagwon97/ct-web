const express = require("express");
const path = require("path");
const postRouter = require('./post.js');
const app = express();


app.use('/', postRouter);
app.use(express.static(path.join(__dirname, "../")));
app.use('/ct__image', express.static(__dirname + "../uploads"));
app.use(express.json());
app.use(express.urlencoded({extended: false}));
app.get('/', (req, res)=>{
  res.status(200).sendFile(path.join(__dirname,"../page/home.html"));
});
app.get('/home', (req, res)=>{
  res.status(200).sendFile(path.join(__dirname,"../page/home.html"));
});
app.get('/inference', (req, res)=>{
  res.status(200).sendFile(path.join(__dirname,"../page/inference.html"));
});
app.get('/reference', (req, res)=>{
  res.status(200).sendFile(path.join(__dirname,"../page/reference.html"));
});



app.listen(3000, ()=>{
  console.log();
})

