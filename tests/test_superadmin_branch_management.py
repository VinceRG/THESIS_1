import unittest
import os
import re
import tempfile

_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), 'superadmin_test.db')
os.environ['DATABASE_URL'] = 'sqlite:///' + _TEST_DB_PATH.replace('\\', '/')

from werkzeug.security import generate_password_hash, check_password_hash
from app import app, db, User, Branch, StaffMember

class SuperadminBranchManagementTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config.update(
            TESTING=True,
            SECRET_KEY='test-secret',
            WTF_CSRF_ENABLED=False,
        )
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

        # Seed main branch
        self.branch = Branch.query.filter_by(code='MAIN').first()
        if not self.branch:
            self.branch = Branch(
                name='Test Main Branch',
                code='MAIN',
                address='123 Test St',
                is_main=True,
                is_active=True,
                operating_status='Open',
                manager_name='Maria Santos',
                head_doctor_name='Dr. Juan Reyes'
            )
            db.session.add(self.branch)
            db.session.commit()
        else:
            self.branch.operating_status = 'Open'
            self.branch.manager_name = 'Maria Santos'
            self.branch.head_doctor_name = 'Dr. Juan Reyes'
            db.session.commit()

        # Seed superadmin
        self.admin = User.query.filter_by(username='superadmin_test_mgr').first()
        if not self.admin:
            self.admin = User(
                username='superadmin_test_mgr',
                password=generate_password_hash('admin123'),
                role='superadmin',
                branch_id=self.branch.id,
                is_active=True,
                must_change_password=False
            )
            db.session.add(self.admin)
        else:
            self.admin.password = generate_password_hash('admin123')
            self.admin.is_active = True
            self.admin.must_change_password = False

        # Seed regular staff/admin
        self.staff_user = User.query.filter_by(username='branch_admin_test_mgr').first()
        if not self.staff_user:
            self.staff_user = User(
                username='branch_admin_test_mgr',
                password=generate_password_hash('staff123'),
                role='branch_admin',
                branch_id=self.branch.id,
                is_active=True,
                must_change_password=False
            )
            db.session.add(self.staff_user)
        else:
            self.staff_user.is_active = True
            self.staff_user.password = generate_password_hash('staff123')
            self.staff_user.must_change_password = False

        db.session.commit()

    def tearDown(self):
        db.session.remove()
        self.app_context.pop()

    @staticmethod
    def _extract_csrf_token(html):
        match = re.search(rb'name="_csrf_token"\s+value="([^"]*)"', html)
        return match.group(1).decode() if match else ''

    def _login(self, username='superadmin_test_mgr', password='admin123'):
        login_page = self.client.get('/login')
        token = self._extract_csrf_token(login_page.data)
        return self.client.post(
            '/login',
            data={'username': username, 'password': password, '_csrf_token': token},
            follow_redirects=True,
        )

    def test_branch_model_fields(self):
        self.assertEqual(self.branch.operating_status, 'Open')
        self.assertEqual(self.branch.manager_name, 'Maria Santos')
        self.assertEqual(self.branch.head_doctor_name, 'Dr. Juan Reyes')

    def test_reset_user_password(self):
        self._login()
        branches_page = self.client.get('/branches')
        token = self._extract_csrf_token(branches_page.data)

        res = self.client.post(
            f'/branches/users/{self.staff_user.id}/reset-password',
            data={'_csrf_token': token},
            follow_redirects=True
        )
        self.assertEqual(res.status_code, 200)

        # Staff user should now have must_change_password = True
        user = db.session.get(User, self.staff_user.id)
        self.assertTrue(user.must_change_password)
        # Old password should no longer match
        self.assertFalse(check_password_hash(user.password, 'staff123'))

    def test_force_password_change(self):
        self._login()
        branches_page = self.client.get('/branches')
        token = self._extract_csrf_token(branches_page.data)

        res = self.client.post(
            f'/branches/users/{self.staff_user.id}/force-password-change',
            data={'_csrf_token': token},
            follow_redirects=True
        )
        self.assertEqual(res.status_code, 200)

        user = db.session.get(User, self.staff_user.id)
        self.assertTrue(user.must_change_password)

    def test_toggle_user_active_and_prevent_self_deactivation(self):
        self._login()
        branches_page = self.client.get('/branches')
        token = self._extract_csrf_token(branches_page.data)

        # Cannot deactivate self
        res_self = self.client.post(
            f'/branches/users/{self.admin.id}/toggle-active',
            data={'_csrf_token': token},
            follow_redirects=True
        )
        self.assertIn(b'cannot deactivate your own', res_self.data.lower())
        admin_user = db.session.get(User, self.admin.id)
        self.assertTrue(admin_user.is_active)

        # Deactivate target staff user
        res_other = self.client.post(
            f'/branches/users/{self.staff_user.id}/toggle-active',
            data={'_csrf_token': token},
            follow_redirects=True
        )
        self.assertEqual(res_other.status_code, 200)
        user = db.session.get(User, self.staff_user.id)
        self.assertFalse(user.is_active)

    def test_deactivated_user_cannot_login(self):
        # Deactivate staff user
        self.staff_user.is_active = False
        db.session.commit()

        login_page = self.client.get('/login')
        token = self._extract_csrf_token(login_page.data)

        # Attempt login
        res = self.client.post('/login', data={
            'username': 'branch_admin_test_mgr',
            'password': 'staff123',
            '_csrf_token': token
        }, follow_redirects=True)
        self.assertIn(b'account has been deactivated', res.data.lower())

    def test_edit_branch_with_staff_dropdown(self):
        self._login()

        # Create sample doctor and manager staff
        doc = StaffMember(
            name='Dr. Test Physician',
            role='General Physicians',
            branch_id=self.branch.id,
            is_active=True
        )
        mgr = StaffMember(
            name='Test Manager Staff',
            role='Registered Medical Technologists',
            branch_id=self.branch.id,
            is_active=True
        )
        db.session.add_all([doc, mgr])
        db.session.commit()

        # Render edit form to verify dropdowns exist
        edit_page = self.client.get(f'/branches/{self.branch.id}/edit')
        self.assertEqual(edit_page.status_code, 200)
        self.assertIn(b'name="manager_id"', edit_page.data)
        self.assertIn(b'name="head_doctor_id"', edit_page.data)
        self.assertIn(b'Dr. Test Physician', edit_page.data)
        self.assertIn(b'Test Manager Staff', edit_page.data)

        # Submit form with staff selections
        res = self.client.post(f'/branches/{self.branch.id}/edit', data={
            'name': 'Updated Branch Name',
            'code': self.branch.code,
            'operating_status': 'Open',
            'manager_id': str(mgr.id),
            'head_doctor_id': str(doc.id),
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        # Verify DB updated with foreign keys and names
        branch = db.session.get(Branch, self.branch.id)
        self.assertEqual(branch.manager_id, mgr.id)
        self.assertEqual(branch.manager_name, 'Test Manager Staff')
        self.assertEqual(branch.head_doctor_id, doc.id)
        self.assertEqual(branch.head_doctor_name, 'Dr. Test Physician')

if __name__ == '__main__':
    unittest.main()
