# php-ext-model

This extension enable you the power of rust and HuggingFace candle framework to run any setence-transformers model form the PHP with lighting speed.

## How to Use

Here's an example of how to use it:

```php
<?php
use Manticore\Ext\Model;
// One from https://huggingface.co/sentence-transformers
$model = Model::create("sentence-transformers/all-MiniLM-L12-v2");
var_dump($model->predict("Hello world"));
```

This will display the flatten tensor as list of floats

## How to Build

You need to have `cargo` installed to build.

```bash
cargo build --release
```

After the build is complete, you can use the extension with PHP as usual.

```bash
$ php -d extension=target/release/libphp_ext_model.so -r 'var_dump(class_exists("Manticore\Ext\Model"));'
bool(true)
```
