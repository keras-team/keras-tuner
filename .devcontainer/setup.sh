sudo pip install --upgrade pip
sudo pip install -e ".[tensorflow-cpu,tests]"
echo "sh shell/lint.sh" > .git/hooks/pre-commit
chmod a+x .git/hooks/pre-commit