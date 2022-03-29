const express = require('express');
const path = require('path')
const multer = require('multer');
const mutler = require('multer');
const md5 = require('md5')
const mongoose = require('mongoose')
const User = require('./model/account')

mongoose.connect('mongodb://localhost/Tumor', () => console.log('...'))

//python shell
const  { PythonShell }  = require('python-shell');

//userDetails
let details= []

//set storage
const storage = mutler.diskStorage({
    destination: './public/uploads/',
    filename: function(req, file, cb){
        cb(null,file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
})

//init upload
const upload = multer({
    storage: storage,
    limits: {fieldSize: 1000000},
    fileFilter: function(req, file, cb){
        checkType(file, cb);
   }
}).single('myImage');

//check file type

checkType = (file, cb) => {
      // allowed extension
      const fileType = /jpeg|jpg|png|gif/; //regular expression 
      //check extension
      const extName = fileType.test(path.extname(file.originalname).toLowerCase());
      //check mimetype
      const mimeType = fileType.test(file.mimetype);

      if( extName && mimeType) {
          cb(null, true);
      } else{
          cb('Error: Images Only')
      }
}

const app = express();

app.use(express.static('./public'));

app.use(express.urlencoded({extended:false}));

app.set('view engine', 'ejs');


// app.get('/index', (req,res) => {
//     res.render('index')
// })

app.get('/about', (req,res) => {
    res.render('about')
})

app.get('/', (req,res)=> {
    res.render('main');
});

app.get('/signup', (req,res) => {
    res.render('signup')
})

app.post('/signup', async (req,res) => {

    const email = req.body.email
    const name = req.body.name
    const p =req.body.password
    const p1 = req.body.repassword
    User.findOne( {email:email}, async (err,person) => {
        if(err){
            console.log(err);
        }
        else{
            if(person){
                res.render('signup', {op:'Email Already Registered'})
            } else{
                if(p===p1){
                let user = new User( {
                    name: req.body.name,
                    email: req.body.email,
                    password: md5(req.body.password)
                })
                try{
                   await user.save()
                } catch{
                    console.log(err);
                }
                
                console.log(user);
                res.render('detail')
            } else{
                res.render('signup', {op:'Password Dosent match',name:name,email:email })
            }
                
                }
        }
    })
})

app.get('/login', (req,res) => {
   res.render('login')
})

app.post('/login', async (req,res) => {

    const email = req.body.email
    const pass = md5(req.body.password)
   // res.render('index')
   User.findOne( {email:email}, (err,person) => {
        if(err){
            console.log(err);
        }
        else{
            if(person){
                if(person.password === pass){
                    res.render('detail')
                } else{
                    res.render('login', {op:'incorrect password'})
                }
            } else{
                res.render('login', {op:'Please Create your Account'})
            }
        }
    })
})

// app.get('/detail', (req,res) => {
//     res.render('detail')
// })

app.post('/detail', (req,res) => {
    const userDetails = {
        name:req.body.name,
        age:req.body.age,
        gender:req.body.gender,
        drName:req.body.drName
    }
    details.push(userDetails)
    console.log(details);
    res.render('index')

})

app.post('/', (req,res) => {
     console.log(details);
    upload(req, res ,(err) => {
        if(err){
            res.render('index', {msg:err })
        } else{
           // console.log(req.file);
           // res.send("ok")
           if(req.file === undefined){
               res.render('index', {msg: 'No file Selected'})
           } else{

            const options = {
                mode: 'text',
                //pythonPath: '/usr/bin/python', 
                pythonOptions: ['-u'],
                // make sure you use an absolute path for scriptPath
                //scriptPath: '//Users/91877/Desktop/nodeProjects/live in lab/',
                args: [req.file.filename]
              };
            
                PythonShell.run('test.py', options, function (err, results) {
                    if (err) throw err;
                    // results is an array consisting of messages collected during execution
                    console.log('results: %j', results);
                    const output = parseInt(results[0].slice(2,3))
                    //const foutput = output[0].slice(2)
                    const i1 = results[1]
                    const i2 = results[2]
                    const i3 = results[3]
                    console.log(output);
                    let finalOutput;
                    output ? finalOutput = 'Present' : finalOutput = 'Absent'
                    res.render('output', {file: `uploads/${req.file.filename}`, finalOutput:finalOutput, details:details, i1:i1, i2:i2, i3:i3})
                    details=[]
                  });
            
               
            //    file= `uploads/${req.file.filename}`;
            //    console.log(file);
           }
        }
    })
    
})



app.listen(3000)