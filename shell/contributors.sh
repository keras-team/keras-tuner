# node shell/generate_json.js
gh api -H "Accept: application/vnd.github+json" /repos/keras-team/keras-tuner/contributors --paginate > response.json
sed "s/\]\[/,/g" response.json > contributors.json
rm response.json
mkdir avatars
python shell/contributors.py avatars > docs/contributors.svg
rm contributors.json
rm -rf avatars
